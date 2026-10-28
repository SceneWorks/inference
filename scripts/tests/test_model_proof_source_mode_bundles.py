"""Frozen-source mode shapes exercised with JSON/process mocks only; no model input IO."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_proof_inputs as inputs
from test_model_proof_inputs import DIGEST, OTHER, READER, event, item, row
import test_model_proof_collector as collector

DIRECTORY = 'mlx-llm::primitives::Weights::from_dir'
CONFIG = 'crates/llm/mlx-llm/src/provider.rs::read_config_value::std::fs::read_to_string@232605'
TOKENIZER = 'core-llm::tokenizer::Tokenizer::from_file'
GGUF = 'mlx-llm::gguf::GgufFile::open'
SIDECAR = 'crates/llm/mlx-llm/src/provider.rs::sidecar_chat_template::std::fs::read_to_string@120781'
JINJA = 'core-llm::template::JinjaChatTemplate::from_tokenizer_config_file/read_to_string'
RGBA = 'crates/media/candle-gen/candle-gen-qwen-image-2-1/tests/e2e_real_weights.rs::transparent_reference::image::open@137967'
SDXL_PAIR = 'candle-gen-sdxl::pipeline::SdxlTokenizers::load/tokenizers'
SDXL_TRAINER_PAIR = 'candle-gen-sdxl::training::DualClip::load/tokenizers'
WRITE_READ_PROOF = {'case_source': 'crates/case/generated.rs', 'case_source_sha256': DIGEST,
                    'writer': 'case writes its own numeric fixture (Array::save_safetensors) then reads it back'}


def generated_case(slots=1):
    value = row('source-defined')
    value['input_contract']['case_generated_inputs'] = True
    value['input_contract']['requirements'] = [
        {**slot('written-fixture-' + str(index), READER, 'model-weights'), 'case_generated_write_read': dict(WRITE_READ_PROOF)}
        for index in range(slots)]
    return value


def slot(name, reader, role, grade='reader-binding'):
    return {'slot': name, 'role': role, 'readers': [reader], 'minimum_collections': 1,
            'minimum_items': 1, 'required_grade': grade, 'scope': 'case'}


def source_case(template=False):
    value = row()
    common = [slot('config', CONFIG, 'artifact-input'), slot('tokenizer', TOKENIZER, 'tokenizer')]
    directory = common + [slot('weights', DIRECTORY, 'model-weights')]
    bundles = ([{'source_identity': 'source.rs::case/sidecar', 'requirements': common + [slot('sidecar', SIDECAR, 'artifact-input')]},
                {'source_identity': 'source.rs::case/jinja', 'requirements': common + [slot('jinja', JINJA, 'tokenizer')]}]
               if template else
               [{'source_identity': 'source.rs::case/directory', 'requirements': directory},
                {'source_identity': 'source.rs::case/direct-gguf', 'requirements': [slot('embedded-model-config-tokenizer', GGUF, 'tensor-input')]}])
    if template:
        bundles[1]['additional_observed_readers'] = [{'reader': SIDECAR, 'role': 'artifact-input'}]
    value['input_contract'].update(requirements=[], requirement_bundles=bundles)
    return value


def observation(reader, path=None, **kwargs):
    return event(item(path or '/MOCK/' + reader.split('::')[-1]), reader=reader, role=inputs.READERS[reader], **kwargs)


def directory_log():
    return observation(CONFIG) + observation(TOKENIZER) + observation(DIRECTORY)


class SourceModeBundles(unittest.TestCase):
    def setUp(self):
        roles = patch.dict(inputs.READERS, {READER: 'model-weights'})
        grades = patch.dict(inputs.READER_GRADES, {READER: frozenset(('reader-binding', 'content', 'validated-content'))})
        roles.start(); grades.start()
        self.addCleanup(roles.stop); self.addCleanup(grades.stop)

    def test_complete_directory_bundle_preserves_actual_atomic_roles(self):
        result = inputs.assess_inputs(source_case(), directory_log())
        self.assertEqual(len(result['actual_inputs']), 3)
        self.assertEqual([value['source_identity'] for value in result['input_requirement_bundles']], ['source.rs::case/directory'])
        self.assertEqual({value['role'] for value in result['actual_inputs']}, {'model-weights', 'artifact-input', 'tokenizer'})

    def test_one_direct_gguf_collection_fulfils_one_combined_bundle(self):
        result = inputs.assess_inputs(source_case(), observation(GGUF))
        self.assertEqual(len(result['actual_inputs']), 1)
        self.assertEqual(len(result['required_input_slots']), 1)
        self.assertEqual(result['actual_inputs'][0]['role'], 'tensor-input')
        self.assertFalse(result['checkpoint_byte_integrity_attested'])
        self.assertFalse(result['terminal_model_evidence'])

    def test_no_primary_only_or_partial_directory_acceptance(self):
        for text in ('', observation(DIRECTORY), observation(DIRECTORY) + observation(CONFIG), observation(TOKENIZER)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                inputs.assess_inputs(source_case(), text)

    def test_reader_role_cartesian_substitution_rejected(self):
        text = event(item(), reader=GGUF, role='model-weights')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(source_case(), text)
        value = source_case()
        value['input_contract']['requirement_bundles'][1]['requirements'][0]['role'] = 'model-weights'
        with self.assertRaises(ValueError):
            inputs.validate_contract(value, required=True)

    def test_source_bundle_cannot_double_count_one_gguf_receipt(self):
        value = source_case()
        original = value['input_contract']['requirement_bundles'][1]['requirements'][0]
        value['input_contract']['requirement_bundles'][1]['requirements'] += [{**original, 'slot': 'duplicate-logical-slot'}]
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, observation(GGUF))

    def test_sidecar_or_jinja_and_sidecar_plus_jinja_fallback_all_observations_survive(self):
        common = observation(CONFIG) + observation(TOKENIZER)
        for suffix, count in ((observation(SIDECAR), 3), (observation(JINJA), 3),
                              (observation(SIDECAR) + observation(JINJA), 4)):
            with self.subTest(count=count):
                result = inputs.assess_inputs(source_case(template=True), common + suffix)
                self.assertEqual(len(result['actual_inputs']), count)
                self.assertEqual(result['input_grade'], 'successful-reader-binding')
                self.assertFalse(result['verified_named_revision'])

    def test_mixed_external_tagged_fallback_observations_are_not_discarded(self):
        value = source_case(template=True); value['input_contract']['fixture_kind'] = 'mixed'
        text = ''
        for reader in (CONFIG, TOKENIZER, SIDECAR, JINJA):
            observed = item('/MOCK/' + reader)
            observed['source_origin'] = 'external-selected'
            text += event(observed, reader=reader, role=inputs.READERS[reader])
        result = inputs.assess_inputs(value, text)
        self.assertEqual(len(result['actual_inputs']), 4)
        self.assertEqual([bundle['source_identity'] for bundle in result['input_requirement_bundles']], ['source.rs::case/jinja'])

    def test_common_requirements_apply_to_every_alternative(self):
        value = source_case()
        value['input_contract']['requirements'] = [slot('common-tokenizer', TOKENIZER, 'tokenizer')]
        # Rename the directory-local tokenizer slot rather than silently reuse its common receipt.
        value['input_contract']['requirement_bundles'][0]['requirements'] = [slot('weights', DIRECTORY, 'model-weights')]
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, observation(GGUF))
        self.assertEqual(len(inputs.assess_inputs(value, observation(GGUF) + observation(TOKENIZER))['actual_inputs']), 2)

    def test_wrong_scope_or_incomplete_extra_observation_rejects_complete_bundle(self):
        for suffix in (observation(CONFIG, scope='tests::other'), observation(CONFIG, complete=False)):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                inputs.assess_inputs(source_case(), observation(GGUF) + suffix)

    def test_unknown_extra_reader_rejects_even_with_complete_bundle(self):
        with self.assertRaises(ValueError):
            inputs.assess_inputs(source_case(), observation(GGUF) + event(item(), reader='unknown::reader'))

    def test_weak_cannot_replace_strong_or_wrong_existing_pin(self):
        value = source_case()
        strong = slot('pinned-source', READER, 'model-weights', 'validated-content')
        strong['expected_sha256'] = [DIGEST]
        value['input_contract']['requirement_bundles'][0]['requirements'] = [strong]
        for observed in (item(), item(kind='validated-content', digest=OTHER)):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                # A valid OTHER bundle must not hide a bad observed strong-source collection.
                inputs.assess_inputs(value, observation(GGUF) + event(observed))
        good = inputs.assess_inputs(value, event(item(kind='validated-content')))
        self.assertTrue(good['required_input_content_attested'])
        self.assertTrue(good['checkpoint_byte_integrity_attested'])
        self.assertFalse(good['verified_named_revision'])

    def test_every_observed_collection_checked_not_only_first_matching_pin(self):
        value = source_case()
        strong = slot('pinned-source', READER, 'model-weights', 'content')
        strong['expected_sha256'] = [DIGEST]
        value['input_contract']['requirement_bundles'][0]['requirements'] = [strong]
        text = event(item(kind='same-read-buffer')) + event(item('/MOCK/other', kind='same-read-buffer', digest=OTHER))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, text)

    def test_extra_weak_observation_cannot_upgrade_content_receipt_or_aggregate(self):
        value = source_case()
        value['input_contract']['requirement_bundles'][0]['requirements'] = [slot('source', READER, 'model-weights', 'content')]
        result = inputs.assess_inputs(value, event(item(kind='same-read-buffer')) + observation(CONFIG))
        self.assertEqual(result['acceptance_scope'], 'source-bound-case-execution')
        self.assertEqual(result['input_grade'], 'successful-reader-binding')
        self.assertFalse(inputs.receipt_claims([result], True)['checkpoint_byte_integrity_attested'])

    def test_empty_duplicate_nested_or_operator_controlled_bundles_reject(self):
        mutations = [[], [{'source_identity': 'empty', 'requirements': []}],
                     [{'source_identity': 'x', 'requirements': [], 'environment': {'MODE': 'gguf'}}]]
        duplicate = source_case()['input_contract']['requirement_bundles'][0]
        mutations.append([duplicate, copy.deepcopy(duplicate)])
        nested = copy.deepcopy(duplicate); nested['requirement_bundles'] = []
        mutations.append([duplicate, nested])
        for bundles in mutations:
            value = source_case(); value['input_contract']['requirement_bundles'] = bundles
            with self.subTest(bundles=bundles), self.assertRaises(ValueError):
                inputs.validate_contract(value, required=True)

    def test_source_defined_or_unresolved_cannot_bypass_with_mode_bundles(self):
        for kind in ('source-defined', 'unresolved'):
            value = source_case(); value['input_contract']['fixture_kind'] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inputs.assess_inputs(value, observation(GGUF))

    def test_full_collector_seal_and_artifact_readback_keep_bundle_and_weak_claims(self):
        value = collector.case(); value['input_contract'] = source_case()['input_contract']
        text = observation(GGUF, scope=value['fqn'])
        helper = collector.Collector('test_full_mock_collector_records_body_device_events_and_safe_upload')
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, _, output = helper._mock_run(Path(temporary).resolve(), row_override=value, input_body=text)
            self.assertEqual(status, 0)
            self.assertEqual(len(receipt['cases'][0]['input_requirement_bundles']), 1)
            self.assertFalse(verified['checkpoint_byte_integrity_attested'])
            receipt['cases'][0]['input_requirement_bundles'] = []
            (output / 'receipt.json').write_text(json.dumps(receipt), encoding="utf-8")
            (output / '00000/result.json').write_text(json.dumps(receipt['cases'][0]), encoding="utf-8")
            with self.assertRaises(ValueError):
                collector.proof.verify_artifacts(output, Path(temporary).resolve(), collector.request(), '1' * 40, (value,))

    def test_positive_specific_mode_cannot_fall_back_to_a_smaller_subset(self):
        value = source_case()
        hadamard = 'crates/llm/mlx-llm/src/prism.rs::from_dir::std::fs::read_to_string@5963'
        specific = copy.deepcopy(value['input_contract']['requirement_bundles'][0])
        specific['source_identity'] = 'source.rs::case/prism-with-pinned-component'
        strong = slot('specific-pinned-component', READER, 'model-weights', 'content')
        strong['expected_sha256'] = [DIGEST]
        specific['requirements'] += [slot('hadamard-signs', hadamard, 'artifact-input'), strong]
        value['input_contract']['requirement_bundles'].append(specific)
        text = directory_log() + observation(hadamard)
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, text)
        result = inputs.assess_inputs(value, text + event(item(kind='same-read-buffer')))
        self.assertEqual([bundle['source_identity'] for bundle in result['input_requirement_bundles']], [specific['source_identity']])
        self.assertEqual(len(result['actual_inputs']), 5)

    def test_supporting_pairs_cannot_fulfil_missing_required_work(self):
        value = source_case()
        value['input_contract']['requirement_bundles'][1]['additional_observed_readers'] = [{'reader': SIDECAR, 'role': 'artifact-input'}]
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, observation(SIDECAR))
        result = inputs.assess_inputs(value, observation(GGUF) + observation(SIDECAR))
        self.assertEqual(len(result['required_input_slots']), 1)
        self.assertEqual(len(result['actual_inputs']), 2)

    def test_supporting_pair_cannot_hide_an_existing_strong_requirement(self):
        value = source_case()
        strong = slot('pinned-source', READER, 'model-weights', 'content')
        strong['expected_sha256'] = [DIGEST]
        value['input_contract']['requirement_bundles'][0]['requirements'] = [strong]
        value['input_contract']['requirement_bundles'][1]['additional_observed_readers'] = [{'reader': READER, 'role': 'model-weights'}]
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, observation(GGUF) + event(item()))

    def test_additional_pairs_are_exact_known_roles_not_operator_authority(self):
        for additional in ([{'reader': GGUF, 'role': 'model-weights'}],
                           [{'reader': 'unknown::reader', 'role': 'tensor-input'}],
                           [{'reader': GGUF, 'role': 'tensor-input', 'environment': {'MODE': 'gguf'}}],
                           [{'reader': GGUF, 'role': 'tensor-input'}] * 2):
            value = source_case()
            value['input_contract']['requirement_bundles'][0]['additional_observed_readers'] = additional
            with self.subTest(additional=additional), self.assertRaises(ValueError):
                inputs.validate_contract(value, required=True)

    def test_source_declared_default_basis_is_not_a_fabricated_read_event(self):
        value = row('mixed')
        value['input_contract']['source_defined_default'] = True
        optional = slot('source-generated-or-external', GGUF, 'tensor-input')
        optional['when'] = 'external-selected'
        value['input_contract']['requirements'].append(optional)
        default = inputs.assess_inputs(value, event(item()))
        self.assertEqual(len(default['actual_inputs']), 1)
        self.assertEqual(default['input_source_branches'][0]['source_origin_basis'], 'source-defined-branch-contract')
        self.assertEqual(default['input_grade'], 'successful-reader-binding', 'mandatory external model input remains external')
        selected = item('/MOCK/override'); selected['source_origin'] = 'external-selected'
        external = inputs.assess_inputs(value, event(item()) + event(selected, reader=GGUF, role='tensor-input'))
        self.assertEqual(external['input_source_branches'][0]['source_origin_basis'], 'actual-reader-tag')
        value['input_contract'].pop('source_defined_default')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, event(item()))

    def test_external_case_optional_reference_default_keeps_mandatory_external_input(self):
        value = row()
        value['input_contract']['source_defined_default'] = True
        optional = slot('optional-rgba-reference', RGBA, 'reference')
        optional['when'] = 'external-selected'
        value['input_contract']['requirements'].append(optional)
        result = inputs.assess_inputs(value, event(item()))
        self.assertEqual(value['input_contract']['fixture_kind'], 'external')
        self.assertEqual(result['input_grade'], 'successful-reader-binding')
        self.assertEqual(result['input_source_branches'][0]['source_origin_basis'], 'source-defined-branch-contract')
        self.assertEqual(len(result['actual_inputs']), 1, 'no default reference receipt is fabricated')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, '')

    def test_external_override_requires_actual_origin_and_every_observation(self):
        value = row(); value['input_contract']['source_defined_default'] = True
        optional = slot('optional-rgba-reference', RGBA, 'reference'); optional['when'] = 'external-selected'
        value['input_contract']['requirements'].append(optional)
        selected = item('/MOCK/reference'); selected['source_origin'] = 'external-selected'
        text = event(item()) + event(selected, reader=RGBA, role='reference')
        result = inputs.assess_inputs(value, text)
        self.assertEqual(result['input_source_branches'][0]['source_origin_basis'], 'actual-reader-tag')
        self.assertEqual(len(result['actual_inputs']), 2)
        unknown = item('/MOCK/unknown-origin')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, event(item()) + event(unknown, reader=RGBA, role='reference'))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, text + event(selected, reader=RGBA, role='reference', complete=False))
        other = item('/MOCK/undeclared'); other['source_origin'] = 'external-selected'
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, text + event(other, reader=CONFIG, role='artifact-input'))

    def test_external_optional_only_flat_or_any_bundle_is_not_a_noinput_escape(self):
        optional = slot('optional-rgba-reference', RGBA, 'reference'); optional['when'] = 'external-selected'
        value = row(); value['input_contract'].update(source_defined_default=True, requirements=[optional])
        with self.assertRaises(ValueError):
            inputs.validate_contract(value, required=True)
        value = source_case(); value['input_contract']['source_defined_default'] = True
        value['input_contract']['requirement_bundles'][0]['requirements'] = [optional]
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, observation(GGUF))

    def test_external_optional_common_slot_keeps_each_source_bundle_unconditional(self):
        value = source_case(); value['input_contract']['source_defined_default'] = True
        optional = slot('optional-rgba-reference', RGBA, 'reference'); optional['when'] = 'external-selected'
        value['input_contract']['requirements'] = [optional]
        for text in (directory_log(), observation(GGUF)):
            with self.subTest(text=text):
                result = inputs.assess_inputs(value, text)
                self.assertNotEqual(result['input_grade'], 'source-defined')
                self.assertEqual(result['input_source_branches'][0]['source_origin_basis'], 'source-defined-branch-contract')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, '')

    def test_optional_external_slot_cannot_demote_existing_strong_primary_requirement(self):
        value = row(grade='content'); value['input_contract']['source_defined_default'] = True
        optional = slot('optional-rgba-reference', RGBA, 'reference'); optional['when'] = 'external-selected'
        value['input_contract']['requirements'].append(optional)
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, event(item()))
        strong = inputs.assess_inputs(value, event(item(kind='same-read-buffer')))
        self.assertEqual(strong['input_grade'], 'same-read-buffer')
        selected = item('/MOCK/reference'); selected['source_origin'] = 'external-selected'
        weak = inputs.assess_inputs(value, event(item(kind='same-read-buffer')) + event(selected, reader=RGBA, role='reference'))
        self.assertEqual(weak['input_grade'], 'successful-reader-binding')
        self.assertFalse(weak['required_input_content_attested'])
        self.assertFalse(weak['checkpoint_byte_integrity_attested'])

    def test_source_declared_ordered_pair_records_accept_exact_lexical_alias(self):
        value = row()
        value['input_contract']['requirements'] = [slot('sdxl-dual-clip-pair', SDXL_PAIR, 'tokenizer')]
        value['input_contract']['requirements'][0]['minimum_items'] = 2
        aliased = event([item('/MOCK/alias/tokenizer.json'), item('/MOCK/alias/tokenizer.json')],
                        reader=SDXL_PAIR, role='tokenizer')
        result = inputs.assess_inputs(value, aliased)
        self.assertEqual(len(result['actual_inputs'][0]['items']), 2)
        self.assertEqual(result['input_grade'], 'successful-reader-binding')
        distinct = event([item('/MOCK/l/tokenizer.json'), item('/MOCK/b/tokenizer.json')],
                         reader=SDXL_PAIR, role='tokenizer')
        self.assertEqual(len(inputs.assess_inputs(value, distinct)['actual_inputs'][0]['items']), 2)

    def test_ordered_record_arity_is_source_derived_not_contract_authored(self):
        base = row()
        base['input_contract']['requirements'] = [slot('sdxl-dual-clip-pair', SDXL_TRAINER_PAIR, 'tokenizer')]
        aliased = event([item('/MOCK/alias/tokenizer.json')] * 2, reader=SDXL_TRAINER_PAIR, role='tokenizer')
        for declared in (1, 3):
            case = copy.deepcopy(base)
            case['input_contract']['requirements'][0]['minimum_items'] = declared
            with self.subTest(declared=declared), self.assertRaises(ValueError):
                inputs.validate_contract(case, required=True)
        base['input_contract']['requirements'][0]['minimum_items'] = 2
        self.assertEqual(len(inputs.assess_inputs(base, aliased)['actual_inputs'][0]['items']), 2)
        for padding in (1, 3):
            text = event([item('/MOCK/alias/tokenizer.json')] * padding, reader=SDXL_TRAINER_PAIR, role='tokenizer')
            with self.subTest(padding=padding), self.assertRaises(ValueError):
                inputs.assess_inputs(base, text)
        distinct_pair = event([item('/MOCK/l/tokenizer.json'), item('/MOCK/b/tokenizer.json')],
                              reader=SDXL_TRAINER_PAIR, role='tokenizer')
        self.assertEqual(len(inputs.assess_inputs(base, distinct_pair)['actual_inputs'][0]['items']), 2)
        mixed = copy.deepcopy(base)
        mixed['input_contract']['requirements'][0]['readers'] = [SDXL_TRAINER_PAIR, TOKENIZER]
        with self.assertRaises(ValueError):
            inputs.validate_contract(mixed, required=True)

    def test_ordered_reader_multiplicity_without_matching_requirement_rejects(self):
        value = row()
        value['input_contract']['requirements'] = [slot('other-work', READER, 'model-weights')]
        aliased = event([item('/MOCK/alias/tokenizer.json')] * 2, reader=SDXL_PAIR, role='tokenizer')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, aliased)
        role_changed = copy.deepcopy(value)
        role_changed['input_contract']['requirements'] = [slot('wrong-role', SDXL_PAIR, 'model-weights')]
        with self.assertRaises(ValueError):
            inputs.validate_contract(role_changed, required=True)

    def test_inapplicable_duplicate_receipt_rejects_even_beside_complete_work(self):
        # A complete ordinary collection plus an ineligible duplicate-bearing receipt: the
        # duplicate cannot borrow a larger bound declared elsewhere, and applicability is
        # per-receipt against reader/role/scope/condition, so the whole observation rejects.
        complete = row()
        complete['input_contract']['requirements'] = [slot('main-work', READER, 'model-weights')]
        complete['input_contract']['requirements'][0]['minimum_items'] = 3
        whole = event([item('/MOCK/a'), item('/MOCK/b'), item('/MOCK/c')])
        stray = event([item('/MOCK/alias/tokenizer.json')] * 2, reader=SDXL_PAIR, role='tokenizer')
        self.assertTrue(inputs.assess_inputs(complete, whole))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(complete, whole + stray)
        conditioned = row('mixed')
        conditioned['input_contract']['source_defined_default'] = True
        optional = slot('external-or-default-pair', SDXL_PAIR, 'tokenizer')
        optional['minimum_items'] = 2
        optional['when'] = 'external-selected'
        conditioned['input_contract']['requirements'] = [slot('always-work', READER, 'model-weights'), optional]
        untagged_pair = event([item('/MOCK/alias/tokenizer.json')] * 2, reader=SDXL_PAIR, role='tokenizer')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(conditioned, event(item()) + untagged_pair)
        selected = item('/MOCK/alias/tokenizer.json'); selected['source_origin'] = 'external-selected'
        result = inputs.assess_inputs(conditioned, event(item()) + event([selected] * 2, reader=SDXL_PAIR, role='tokenizer'))
        self.assertEqual(result['input_source_branches'][0]['source_origin'], 'external-selected')

    def test_one_ordered_receipt_cannot_be_replayed_across_collections(self):
        value = row()
        pair = slot('pair-collection', SDXL_PAIR, 'tokenizer'); pair['minimum_items'] = 2
        pair['minimum_collections'] = 2
        value['input_contract']['requirements'] = [pair]
        single = event([item('/MOCK/alias/tokenizer.json')] * 2, reader=SDXL_PAIR, role='tokenizer')
        for text in (single, single + single):
            with self.assertRaises(ValueError):
                inputs.assess_inputs(value, text)
        both = single + event([item('/MOCK/other/tokenizer.json')] * 2, reader=SDXL_PAIR, role='tokenizer')
        result = inputs.assess_inputs(value, both)
        self.assertEqual(len(result['actual_inputs']), 2)

    def test_non_ordered_readers_keep_strict_distinct_item_identities(self):
        value = row()
        value['input_contract']['requirements'][0]['minimum_items'] = 2
        duplicated = event([item(), item()], reader=READER, role='model-weights')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, duplicated)

    def test_one_ordered_receipt_cannot_satisfy_two_distinct_required_collections(self):
        value = row()
        first = slot('preview-pair', SDXL_PAIR, 'tokenizer'); first['minimum_items'] = 2
        second = slot('trainer-pair', SDXL_PAIR, 'tokenizer'); second['minimum_items'] = 2
        value['input_contract']['requirements'] = [first, second]
        single = event([item('/MOCK/alias/tokenizer.json')] * 2, reader=SDXL_PAIR, role='tokenizer')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, single)
        both = single + event([item('/MOCK/other/tokenizer.json')] * 2, reader=SDXL_PAIR, role='tokenizer')
        result = inputs.assess_inputs(value, both)
        self.assertEqual(len(result['actual_inputs']), 2)

    def test_case_generated_roundtrip_reads_stay_source_bound_under_explicit_flag(self):
        result = inputs.assess_inputs(generated_case(), event(item()))
        self.assertEqual(result['input_grade'], 'source-defined')
        self.assertEqual(result['acceptance_scope'], 'source-bound-case-execution')
        self.assertEqual(len(result['required_input_slots']), 1)
        self.assertFalse(result['checkpoint_byte_integrity_attested'])
        self.assertFalse(result['verified_named_revision'])
        self.assertFalse(result['terminal_model_evidence'])
        with self.assertRaises(ValueError):
            inputs.assess_inputs(generated_case(), '')

    def test_case_generated_flag_requires_exact_per_slot_write_then_read_anchors(self):
        missing = generated_case(); missing['input_contract']['requirements'][0].pop('case_generated_write_read')
        with self.assertRaises(ValueError):
            inputs.validate_contract(missing, required=True)
        partial = generated_case(slots=2); partial['input_contract']['requirements'][1].pop('case_generated_write_read')
        with self.assertRaises(ValueError):
            inputs.validate_contract(partial, required=True)
        for field, value in (('case_source', ''), ('case_source_sha256', 'not-a-sha'),
                             ('case_source_sha256', '0' * 64), ('writer', '')):
            bad = generated_case(); bad['input_contract']['requirements'][0]['case_generated_write_read'] = {
                **WRITE_READ_PROOF, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                inputs.validate_contract(bad, required=True)
        extra = generated_case(); extra['input_contract']['requirements'][0]['case_generated_write_read']['family'] = 'inference'
        with self.assertRaises(ValueError):
            inputs.validate_contract(extra, required=True)

    def test_case_generated_flag_is_source_defined_only_and_not_a_general_bypass(self):
        unflagged = generated_case(); unflagged['input_contract'].pop('case_generated_inputs')
        with self.assertRaises(ValueError):
            inputs.validate_contract(unflagged, required=True)
        anchored_without_flag = row('source-defined')
        anchored_without_flag['input_contract']['requirements'] = [
            {**slot('written-fixture', READER, 'model-weights'), 'case_generated_write_read': dict(WRITE_READ_PROOF)}]
        with self.assertRaises(ValueError):
            inputs.validate_contract(anchored_without_flag, required=True)
        plain_without_flag = row('source-defined')
        plain_without_flag['input_contract']['requirements'] = [slot('plain-slot', READER, 'model-weights')]
        with self.assertRaises(ValueError):
            inputs.validate_contract(plain_without_flag, required=True)
        anchored_external = row()
        anchored_external['input_contract']['requirements'] = [
            {**slot('anchored-external-slot', READER, 'model-weights'), 'case_generated_write_read': dict(WRITE_READ_PROOF)}]
        with self.assertRaises(ValueError):
            inputs.validate_contract(anchored_external, required=True)
        for kind in ('external', 'mixed', 'unresolved'):
            wrong_kind = generated_case(); wrong_kind['input_contract']['fixture_kind'] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inputs.validate_contract(wrong_kind, required=True)
        unflagged_empty = row('source-defined')
        self.assertEqual(inputs.assess_inputs(unflagged_empty, '')['input_grade'], 'source-defined')

    def test_case_generated_receipt_binds_exactly_one_declared_slot(self):
        two = generated_case(slots=2)
        single = event(item('/MOCK/generated/one.safetensors'))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(two, single)
        both = single + event(item('/MOCK/generated/two.safetensors'))
        result = inputs.assess_inputs(two, both)
        self.assertEqual({slot['slot'] for slot in result['required_input_slots']},
                         {'written-fixture-0', 'written-fixture-1'})
        self.assertEqual(result['input_grade'], 'source-defined')

    def test_external_present_binds_the_obligation_to_the_actual_observation(self):
        value = row()
        golden = slot('optional-golden', GGUF, 'tensor-input')
        golden['when'] = 'external-present'
        value['input_contract']['requirements'].append(golden)
        absent = inputs.assess_inputs(value, event(item()))
        self.assertEqual(absent['input_source_branches'][-1]['source_origin'], 'external-absent')
        self.assertEqual({matched['slot'] for matched in absent['required_input_slots']},
                         {value['input_contract']['requirements'][0]['slot']})
        present = inputs.assess_inputs(value, event(item()) + observation(GGUF))
        self.assertEqual(present['input_source_branches'][-1]['source_origin'], 'external-present')
        self.assertIn('optional-golden', {matched['slot'] for matched in present['required_input_slots']})
        self.assertEqual(len(present['actual_inputs']), 2)

    def test_observed_external_present_input_must_fully_satisfy_its_slot(self):
        value = row()
        golden = slot('optional-golden', GGUF, 'tensor-input')
        golden['when'] = 'external-present'
        value['input_contract']['requirements'].append(golden)
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, event(item()) + event(item(), reader=GGUF, role='model-weights'))
        partial = copy.deepcopy(value)
        partial['input_contract']['requirements'][1]['minimum_items'] = 2
        single = event(item('/MOCK/golden/single'), reader=GGUF, role='tensor-input')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(partial, event(item()) + single)

    def test_external_present_never_weakens_unconditional_or_becomes_a_default(self):
        only = row()
        golden = slot('optional-golden', GGUF, 'tensor-input')
        golden['when'] = 'external-present'
        only['input_contract']['requirements'] = [golden]
        with self.assertRaises(ValueError):
            inputs.validate_contract(only, required=True)
        for kind in ('source-defined',):
            wrong = copy.deepcopy(only); wrong['input_contract']['fixture_kind'] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inputs.validate_contract(wrong, required=True)
        defaulted = row()
        defaulted['input_contract']['source_defined_default'] = True
        golden_default = copy.deepcopy(golden)
        defaulted['input_contract']['requirements'] = [slot('always', READER, 'model-weights'), golden_default]
        with self.assertRaises(ValueError):
            inputs.validate_contract(defaulted, required=True)
        child = copy.deepcopy(only); child.update(execution_role='indexed_children', expected_child_ids=['0'])
        child['input_contract']['requirements'][0]['scope'] = 'each-child'
        with self.assertRaises(ValueError):
            inputs.validate_contract(child, required=True)
        bundle_only = source_case()
        conditional = slot('optional-golden', GGUF, 'tensor-input')
        conditional['when'] = 'external-present'
        bundle_only['input_contract']['requirement_bundles'][0]['requirements'] = [conditional]
        bundle_only['input_contract']['requirement_bundles'][1]['requirements'] = [conditional]
        with self.assertRaises(ValueError):
            inputs.assess_inputs(bundle_only, directory_log() + observation(CONFIG) + observation(TOKENIZER) + observation(GGUF))


class ConditionalOnly(unittest.TestCase):
    def setUp(self):
        roles = patch.dict(inputs.READERS, {READER: 'model-weights'})
        grades = patch.dict(inputs.READER_GRADES, {READER: frozenset(('reader-binding', 'content', 'validated-content'))})
        roles.start(); grades.start()
        self.addCleanup(roles.stop); self.addCleanup(grades.stop)

    def case(self):
        value = row()
        value['input_contract'].update(conditional_only=True, requirements=[
            {**slot('optional-golden', READER, 'model-weights'), 'when': 'external-present'}])
        return value

    def test_absent_is_no_input_not_successful_case_execution_or_content(self):
        result = inputs.assess_inputs(self.case(), '')
        self.assertEqual(result['input_grade'], 'no-input-executed')
        self.assertEqual(result['acceptance_scope'], 'no-input-executed')
        self.assertEqual(result['input_source_branches'][0]['source_origin'], 'external-absent')
        for field in ('required_input_content_attested', 'checkpoint_byte_integrity_attested',
                      'verified_named_revision', 'terminal_model_evidence'):
            self.assertFalse(result[field])
        aggregate = inputs.receipt_claims([result], True)
        self.assertEqual(aggregate['input_grade'], 'no-input-executed')
        self.assertEqual(aggregate['acceptance_scope'], 'unaccepted-incomplete')
        self.assertFalse(aggregate['checkpoint_byte_integrity_attested'])

    def test_present_grades_normally_and_never_claims_terminal_or_named_revision(self):
        result = inputs.assess_inputs(self.case(), event(item()))
        self.assertEqual(result['input_grade'], 'successful-reader-binding')
        self.assertEqual(result['acceptance_scope'], 'source-bound-case-execution')
        self.assertEqual(len(result['required_input_slots']), 1)
        self.assertFalse(result['verified_named_revision'])
        self.assertFalse(result['terminal_model_evidence'])
        partial = self.case(); partial['input_contract']['requirements'][0]['minimum_items'] = 2
        with self.assertRaises(ValueError):
            inputs.assess_inputs(partial, event(item()))

    def test_marker_rejects_unconditional_or_default_or_non_external_and_false(self):
        for kind in ('mixed', 'source-defined', 'unresolved'):
            value = self.case(); value['input_contract']['fixture_kind'] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inputs.validate_contract(value, required=True)
        unconditional = self.case()
        unconditional['input_contract']['requirements'].append(slot('always', GGUF, 'tensor-input'))
        default = self.case(); default['input_contract']['source_defined_default'] = False
        false = self.case(); false['input_contract']['conditional_only'] = False
        empty = self.case(); empty['input_contract']['requirements'] = []
        for value in (unconditional, default, false, empty):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inputs.validate_contract(value, required=True)

    def test_collector_absent_fails_present_passes_and_cannot_relabel_absent_artifact(self):
        for present in (False, True):
            value, text = collector.external_case()
            value['input_contract']['conditional_only'] = True
            value['input_contract']['requirements'][0]['when'] = 'external-present'
            helper = collector.Collector('test_full_mock_collector_records_body_device_events_and_safe_upload')
            with self.subTest(present=present), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                status, receipt, verified, _, output = helper._mock_run(
                    root, row_override=value, input_body=text if present else '')
                self.assertEqual(status, 0 if present else 1)
                self.assertEqual(receipt['accepted'], present)
                self.assertEqual(verified['accepted_case_execution'], present)
                self.assertEqual(receipt['cases'][0]['accepted'], present)
                if not present:
                    self.assertEqual(receipt['cases'][0]['input_grade'], 'no-input-executed')
                    self.assertIn('no input executed', receipt['cases'][0]['error'])
                    receipt.update(accepted=True, completed=1)
                    case = receipt['cases'][0]
                    case['accepted'] = True
                    (output / 'receipt.json').write_text(json.dumps(receipt), encoding="utf-8")
                    (output / '00000/result.json').write_text(json.dumps(case), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        collector.proof.verify_artifacts(output, root, collector.request(), '1' * 40, (value,))

    def test_generated_slots_beside_external_work_keep_both_obligations(self):
        for kind in ('mixed', 'external'):
            value = generated_case()
            value['input_contract']['fixture_kind'] = kind
            value['input_contract']['requirements'].append(slot('real-external', GGUF, 'tensor-input'))
            inputs.validate_contract(value, required=True)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inputs.assess_inputs(value, observation(GGUF))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inputs.assess_inputs(value, event(item()))
            result = inputs.assess_inputs(value, event(item()) + observation(GGUF))
            self.assertEqual(len(result['required_input_slots']), 2)
            self.assertEqual(result['input_grade'], 'successful-reader-binding')
            self.assertFalse(result['terminal_model_evidence'])
            missing_anchor = copy.deepcopy(value)
            missing_anchor['input_contract']['requirements'][0]['case_generated_write_read'] = None
            with self.assertRaises(ValueError):
                inputs.validate_contract(missing_anchor, required=True)
            only_generated = generated_case(); only_generated['input_contract']['fixture_kind'] = kind
            with self.assertRaises(ValueError):
                inputs.validate_contract(only_generated, required=True)
            no_generated = row(kind); no_generated['input_contract']['case_generated_inputs'] = True
            with self.assertRaises(ValueError):
                inputs.validate_contract(no_generated, required=True)

    def test_generated_unconditional_cannot_replace_external_unconditional_work(self):
        value = generated_case(); value['input_contract']['fixture_kind'] = 'external'
        value['input_contract']['requirements'].append({**slot('optional-external', GGUF, 'tensor-input'),
                                                      'when': 'external-present'})
        with self.assertRaises(ValueError):
            inputs.validate_contract(value, required=True)

    def generated_alternatives(self):
        value = row('mixed')
        value['input_contract'].update(case_generated_inputs=True,
            requirements=[slot('common-external', GGUF, 'tensor-input')],
            requirement_bundles=[
                {'source_identity': 'source.rs::preexisting',
                 'requirements': [slot('preexisting-tokenizer', TOKENIZER, 'tokenizer')]},
                {'source_identity': 'source.rs::assembled',
                 'requirements': [{**slot('written-snapshot', READER, 'model-weights'),
                                   'case_generated_write_read': dict(WRITE_READ_PROOF)}]}])
        return value

    def test_generated_union_allows_an_ordinary_arm_and_keeps_generated_arm_obligations(self):
        value = self.generated_alternatives()
        inputs.validate_contract(value, required=True)
        ordinary = inputs.assess_inputs(value, observation(GGUF) + observation(TOKENIZER))
        self.assertEqual([x['source_identity'] for x in ordinary['input_requirement_bundles']],
                         ['source.rs::preexisting'])
        assembled = inputs.assess_inputs(value, observation(GGUF) + event(item()))
        self.assertEqual([x['source_identity'] for x in assembled['input_requirement_bundles']],
                         ['source.rs::assembled'])
        self.assertEqual(len(assembled['required_input_slots']), 2)
        self.assertEqual(assembled['input_grade'], 'successful-reader-binding')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, observation(GGUF))
        missing_anchor = copy.deepcopy(value)
        missing_anchor['input_contract']['requirement_bundles'][1]['requirements'][0]['case_generated_write_read'] = None
        with self.assertRaises(ValueError):
            inputs.validate_contract(missing_anchor, required=True)

    def test_generated_union_rejects_all_generated_arm_and_zero_generated_union(self):
        all_generated = self.generated_alternatives()
        all_generated['input_contract']['requirements'] = []
        with self.assertRaises(ValueError):
            inputs.validate_contract(all_generated, required=True)
        no_generated = self.generated_alternatives()
        no_generated['input_contract']['requirement_bundles'][1]['requirements'][0].pop('case_generated_write_read')
        with self.assertRaises(ValueError):
            inputs.validate_contract(no_generated, required=True)
        external = self.generated_alternatives(); external['input_contract']['fixture_kind'] = 'external'
        external['input_contract']['requirements'][0]['when'] = 'external-present'
        with self.assertRaises(ValueError):
            inputs.validate_contract(external, required=True)

    def disjoint_optionals(self, kind='external'):
        value = row(kind)
        value['input_contract']['source_defined_default'] = True
        value['input_contract']['requirements'] += [
            {**slot('optional-metadata', GGUF, 'tensor-input'), 'when': 'external-present'},
            {**slot('optional-user-tokenizer', TOKENIZER, 'tokenizer'), 'when': 'external-selected'}]
        return value

    def test_disjoint_presence_and_default_slots_keep_their_own_rules(self):
        value = self.disjoint_optionals()
        selected = item('/MOCK/user-tokenizer'); selected['source_origin'] = 'external-selected'
        optional = observation(GGUF)
        external = event(selected, reader=TOKENIZER, role='tokenizer')
        for log in ('', optional, external, optional + external):
            result = inputs.assess_inputs(value, event(item()) + log)
            self.assertEqual(result['input_grade'], 'successful-reader-binding')
            self.assertFalse(result['terminal_model_evidence'])
            self.assertEqual(len(result['actual_inputs']), 1 + bool(optional and optional in log) + bool(external and external in log))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, optional + external)
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, event(item()) + event(item(), reader=TOKENIZER, role='tokenizer'))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, event(item()) + event(item(), reader=GGUF, role='tokenizer'))

    def test_optional_default_and_presence_overlap_rejects_and_noinput_is_capped(self):
        value = self.disjoint_optionals()
        overlap = copy.deepcopy(value)
        overlap['input_contract']['requirements'][1].update(readers=[TOKENIZER], role='tokenizer')
        with self.assertRaises(ValueError):
            inputs.validate_contract(overlap, required=True)
        noinput = self.disjoint_optionals('mixed')
        noinput['input_contract']['requirements'] = noinput['input_contract']['requirements'][1:]
        result = inputs.assess_inputs(noinput, '')
        self.assertEqual(result['input_grade'], 'no-input-executed')
        self.assertFalse(result['checkpoint_byte_integrity_attested'])
        self.assertEqual(inputs.receipt_claims([result], True)['acceptance_scope'], 'unaccepted-incomplete')
        forbidden = copy.deepcopy(noinput); forbidden['input_contract']['conditional_only'] = True
        with self.assertRaises(ValueError):
            inputs.validate_contract(forbidden, required=True)

    def cardinality_modes(self):
        value = row()
        value['input_contract']['requirements'] = []
        value['input_contract']['requirement_bundles'] = [
            {'source_identity': 'source.rs::ordinary', 'requirements': [slot('ordinary-read', READER, 'model-weights')]},
            {'source_identity': 'source.rs::stronger', 'requirements': [
                {**slot('two-reads', READER, 'model-weights'), 'minimum_collections': 2},
                slot('mode-config', CONFIG, 'artifact-input')]}]
        return value

    def test_activated_partial_cannot_hide_in_a_complete_ordinary_arm(self):
        value = self.cardinality_modes()
        ordinary = inputs.assess_inputs(value, event(item('/MOCK/one')))
        self.assertEqual([x['source_identity'] for x in ordinary['input_requirement_bundles']], ['source.rs::ordinary'])
        with self.assertRaisesRegex(ValueError, 'activated.*partial'):
            inputs.assess_inputs(value, event(item('/MOCK/one')) + event(item('/MOCK/two')))
        complete = inputs.assess_inputs(value, event(item('/MOCK/one')) + event(item('/MOCK/two')) + observation(CONFIG))
        self.assertEqual([x['source_identity'] for x in complete['input_requirement_bundles']], ['source.rs::stronger'])
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, '')

    def test_nested_full_modes_pass_disjoint_full_modes_reject_and_identical_need_witness(self):
        nested = self.cardinality_modes()
        nested['input_contract']['requirement_bundles'][1]['requirements'].pop()
        result = inputs.assess_inputs(nested, event(item('/MOCK/one')) + event(item('/MOCK/two')))
        self.assertEqual(len(result['input_requirement_bundles']), 2)
        disjoint = row()
        disjoint['input_contract']['requirements'] = [slot('shared-weights', DIRECTORY, 'model-weights')]
        disjoint['input_contract']['requirement_bundles'] = [
            {'source_identity': 'source.rs::config', 'requirements': [slot('config', CONFIG, 'artifact-input')]},
            {'source_identity': 'source.rs::tokenizer', 'requirements': [slot('tokenizer', TOKENIZER, 'tokenizer')]}]
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            inputs.assess_inputs(disjoint, observation(DIRECTORY) + observation(CONFIG) + observation(TOKENIZER))
        pinned = row()
        pinned['input_contract']['requirements'] = []
        pinned['input_contract']['requirement_bundles'] = [
            {'source_identity': 'source.rs::pin-A', 'requirements': [
                {**slot('pin-A', READER, 'model-weights', 'content'), 'expected_sha256': [DIGEST]}]},
            {'source_identity': 'source.rs::pin-B', 'requirements': [
                {**slot('pin-B', READER, 'model-weights', 'content'), 'expected_sha256': [OTHER]}]}]
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            inputs.assess_inputs(pinned, event(item('/MOCK/A', 'same-read-buffer', DIGEST))
                                 + event(item('/MOCK/B', 'same-read-buffer', OTHER)))
        identical = copy.deepcopy(disjoint)
        identical['input_contract']['requirement_bundles'][1]['requirements'] = [slot('same-config-other-label', CONFIG, 'artifact-input')]
        with self.assertRaisesRegex(ValueError, 'producer witness'):
            inputs.assess_inputs(identical, observation(DIRECTORY) + observation(CONFIG))

    def test_sdk_shared_weights_and_nested_family_config_never_activate_unselected_modes(self):
        value = row()
        call_tokenizer = 'crates/llm/mlx-llm/src/provider.rs::load_admitted::Tokenizer::from_file@87862'
        hadamard = 'crates/llm/mlx-llm/src/prism.rs::from_dir::std::fs::read_to_string@5963'
        joy = 'crates/llm/mlx-llm/src/joycaption.rs::from_dir::std::fs::read_to_string@10743'
        llama = [slot('config', CONFIG, 'artifact-input'), slot('tokenizer', TOKENIZER, 'tokenizer'),
                 slot('callsite-tokenizer', call_tokenizer, 'tokenizer')]
        value['input_contract'].update(requirements=[slot('common-weights', DIRECTORY, 'model-weights')],
            requirement_bundles=[{'source_identity': 'sdk::ordinary', 'requirements': llama},
                {'source_identity': 'sdk::prism', 'requirements': llama + [slot('hadamard', hadamard, 'artifact-input')]},
                {'source_identity': 'sdk::joycaption', 'requirements': [slot('joy-config', joy, 'artifact-input'),
                                                                      slot('tokenizer', TOKENIZER, 'tokenizer')]}])
        text = observation(DIRECTORY) + observation(CONFIG) + observation(TOKENIZER) + observation(call_tokenizer)
        result = inputs.assess_inputs(value, text)
        self.assertEqual([x['source_identity'] for x in result['input_requirement_bundles']], ['sdk::ordinary'])
        prism = inputs.assess_inputs(value, text + observation(hadamard))
        self.assertEqual([x['source_identity'] for x in prism['input_requirement_bundles']], ['sdk::prism'])
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value, observation(DIRECTORY) + observation(hadamard))

    def test_supporting_attempt_never_activates_or_satisfies_a_requirement(self):
        value = source_case(template=True)
        text = observation(CONFIG) + observation(TOKENIZER) + observation(SIDECAR) + observation(JINJA)
        result = inputs.assess_inputs(value, text)
        self.assertEqual([x['source_identity'] for x in result['input_requirement_bundles']], ['source.rs::case/jinja'])
        side = next(x for x in result['input_bundle_requirement_classes'] if x['source_identity']=='source.rs::case/sidecar')
        self.assertFalse(side['activated'])
        bad = copy.deepcopy(value)
        bad['input_contract']['requirement_bundles'][1]['additional_observed_readers'].append(
            {'reader': JINJA, 'role': 'tokenizer'})
        with self.assertRaisesRegex(ValueError, 'supporting.*mandatory'):
            inputs.validate_contract(bad, required=True)

    def test_supporting_mode_probe_does_not_activate_an_incomplete_other_arm(self):
        value = row()
        value['input_contract'].update(requirements=[slot('shared-read', READER, 'model-weights')],
            requirement_bundles=[
                {'source_identity': 'source.rs::ordinary-with-probe',
                 'requirements': [slot('ordinary-second-read', READER, 'model-weights')],
                 'additional_observed_readers': [{'reader': GGUF, 'role': 'tensor-input'}]},
                {'source_identity': 'source.rs::other',
                 'requirements': [slot('mode-GGUF', GGUF, 'tensor-input'),
                                  slot('mode-tokenizer', TOKENIZER, 'tokenizer')]}])
        result = inputs.assess_inputs(value, event(item('/MOCK/one')) + event(item('/MOCK/two')) + observation(GGUF))
        self.assertEqual([x['source_identity'] for x in result['input_requirement_bundles']], ['source.rs::ordinary-with-probe'])
        other = next(x for x in result['input_bundle_requirement_classes'] if x['source_identity']=='source.rs::other')
        self.assertFalse(other['activated'])
        self.assertFalse(other['fully_satisfied'])
        self.assertEqual(len(result['required_input_slots']), 2)

    def test_aggregate_shared_collection_floor_four_to_seven_and_generated_equivalent(self):
        for generated in (False, True):
            value = row('mixed' if generated else 'external')
            value['input_contract']['requirements'] = [slot('external-common', GGUF, 'tensor-input')] if generated else []
            arms = []
            for name, count in (('dense', 4), ('packed', 7)):
                requirements = [slot(name + '-' + str(i), READER, 'model-weights') for i in range(count)]
                if generated:
                    for requirement in requirements:
                        requirement['case_generated_write_read'] = dict(WRITE_READ_PROOF)
                arms.append({'source_identity': 'source.rs::' + name, 'requirements': requirements})
            value['input_contract']['requirement_bundles'] = arms
            if generated:
                value['input_contract']['case_generated_inputs'] = True
            base = observation(GGUF) if generated else ''
            for count in (3, 4, 5, 6, 7):
                text = base + ''.join(event(item('/MOCK/collection-' + str(i))) for i in range(count))
                with self.subTest(generated=generated, count=count):
                    if count in (3, 5, 6):
                        with self.assertRaisesRegex(ValueError, 'activated.*partial' if count in (5, 6) else 'no complete'):
                            inputs.assess_inputs(value, text)
                    else:
                        result = inputs.assess_inputs(value, text)
                        self.assertEqual([x['source_identity'] for x in result['input_requirement_bundles']],
                                         ['source.rs::dense'] if count == 4 else ['source.rs::dense', 'source.rs::packed'])
                        self.assertEqual(result['input_grade'], 'successful-reader-binding')
                        self.assertFalse(result['terminal_model_evidence'])

    def test_same_reader_original_pins_activate_only_the_matching_arm(self):
        value = row()
        value['input_contract']['requirements'] = []
        value['input_contract']['requirement_bundles'] = [
            {'source_identity': 'source.rs::A', 'requirements': [
                {**slot('pin-A', READER, 'model-weights', 'content'), 'expected_sha256': [DIGEST]}]},
            {'source_identity': 'source.rs::B', 'requirements': [
                {**slot('pin-B', READER, 'model-weights', 'content'), 'expected_sha256': [OTHER]}]}]
        result = inputs.assess_inputs(value, event(item('/MOCK/A', 'same-read-buffer', DIGEST)))
        self.assertEqual([x['source_identity'] for x in result['input_requirement_bundles']], ['source.rs::A'])

    def test_nested_implication_never_reuses_one_pinned_collection_unit(self):
        a = {**slot('A', READER, 'model-weights', 'content'), 'expected_sha256': [DIGEST]}
        b = {**slot('B', READER, 'model-weights', 'content'), 'expected_sha256': [OTHER]}
        self.assertFalse(inputs._specific_sets_nested([{**a, 'minimum_collections': 2}], [a, b]))
        self.assertTrue(inputs._specific_sets_nested([a], [{**a, 'minimum_collections': 2}]))

    def residual_case(self, minimum=2, optional_grade='reader-binding'):
        value = row()
        value['input_contract']['requirements'] = [
            {**slot('required', READER, 'model-weights'), 'minimum_collections': minimum},
            {**slot('optional', READER, 'model-weights', optional_grade), 'when': 'external-present'}]
        return value

    def test_residual_distinct_optional_absent_duplicate_and_third_receipt(self):
        value = self.residual_case()
        one = event(item('/MOCK/required-1')); two = event(item('/MOCK/required-2'))
        for text in (one + two, one + two + one):
            result = inputs.assess_inputs(value, text)
            self.assertEqual(len(result['required_input_slots']), 1)
            self.assertEqual(result['input_source_branches'][-1]['source_origin'], 'not-observed')
            self.assertTrue(result['input_source_branches'][-1]['selection_not_attested'])
        present = inputs.assess_inputs(value, one + two + event(item('/MOCK/optional')))
        self.assertEqual(len(present['required_input_slots']), 2)
        self.assertEqual(present['input_source_branches'][-1]['source_origin'], 'external-present')
        self.assertIn('optional I/O failure versus absence', ' '.join(present['input_claim_limits']))
        self.assertFalse(present['terminal_model_evidence'])

    def test_residual_partial_grade_and_overlapping_collection_fail(self):
        value = self.residual_case(minimum=1)
        partial = copy.deepcopy(value); partial['input_contract']['requirements'][1]['minimum_collections'] = 2
        with self.assertRaisesRegex(ValueError, 'residual.*partial'):
            inputs.assess_inputs(partial, event(item('/MOCK/required')) + event(item('/MOCK/optional')))
        strong = self.residual_case(minimum=1, optional_grade='content')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(strong, event(item('/MOCK/required')) + event(item('/MOCK/weak-residual')))
        with self.assertRaisesRegex(ValueError, 'overlaps required'):
            inputs.assess_inputs(value, event(item('/MOCK/required'))
                                 + event([item('/MOCK/required'), item('/MOCK/new')]))

    def test_residual_matching_reserves_strong_optional_independent_of_log_order(self):
        value = self.residual_case(minimum=1, optional_grade='content')
        strong = event(item('/MOCK/optional', 'same-read-buffer', DIGEST))
        weak = event(item('/MOCK/required'))
        for text in (strong + weak, weak + strong):
            result = inputs.assess_inputs(value, text)
            optional = next(x for x in result['required_input_slots'] if x['slot']=='optional')
            self.assertEqual(result['actual_inputs'][optional['receipt_indices'][0]]['items'][0]['provenance']['kind'],
                             'same-read-buffer')
            self.assertEqual(result['input_grade'], 'successful-reader-binding')

    def test_h3_exact_two_case_three_pair_collision_shapes(self):
        config_reader = 'gen-core::model_input_proof::read_configuration_to_string'
        patcher = patch.dict(inputs.READERS, {config_reader: 'configuration'})
        grade_patcher = patch.dict(inputs.READER_GRADES, {config_reader: frozenset(('reader-binding',))})
        with patcher, grade_patcher:
            definitions = [('veda_ab::veda_ab_render_one_cell', [('mlx-gen::Weights::from_file', 'model-weights')]),
                           ('quant_tiers_real::tier_renders_coherently_and_is_close_to_bf16',
                            [('mlx-gen::Weights::from_paths', 'model-weights'), (config_reader, 'configuration')])]
            for fqn, pairs in definitions:
                value = row(); value.update(package='mlx-gen-minimax-h3', fqn=fqn)
                requirements=[];required='';optional=''
                for i,(reader,role) in enumerate(pairs):
                    requirements += [slot('primary-'+str(i), reader, role),
                                     {**slot('optional-'+str(i), reader, role), 'when':'external-present'}]
                    required += event(item('/MOCK/primary-'+str(i)), scope=fqn, reader=reader, role=role)
                    optional += event(item('/MOCK/optional-'+str(i)), scope=fqn, reader=reader, role=role)
                value['input_contract']['requirements']=requirements
                absent=inputs.assess_inputs(value,required)
                self.assertEqual(len(absent['required_input_slots']),len(pairs))
                present=inputs.assess_inputs(value,required+optional)
                self.assertEqual(len(present['required_input_slots']),2*len(pairs))

    def test_residual_same_item_different_provenance_is_still_not_optional_presence(self):
        value = self.residual_case(minimum=1)
        text = event(item('/MOCK/same')) + event(item('/MOCK/same', 'same-read-buffer', DIGEST))
        result = inputs.assess_inputs(value, text)
        self.assertEqual(len(result['actual_inputs']), 2)
        self.assertEqual(len(result['required_input_slots']), 1)
        self.assertEqual(result['input_source_branches'][-1]['source_origin'], 'not-observed')
        self.assertEqual(result['input_grade'], 'successful-reader-binding')

    def test_joy_support_only_llava_cannot_poison_required_ordinary_prism_sharing(self):
        weights='candle-llm::primitives::Weights::from_dir'
        config='crates/llm/candle-llm/src/provider.rs::read_json::std::fs::read_to_string@267613'
        prism='crates/llm/candle-llm/src/prism_checkpoint.rs::read_metadata::std::fs::read@12764'
        llava='crates/llm/candle-llm/src/llava.rs::from_dir_with::std::fs::read_to_string@11509'
        star='crates/llm/candle-llm/src/starvector.rs::read_config::std::fs::read_to_string@10042'
        value=row();value.update(package='candle-gen-joycaption',fqn='conformance::joycaption_conformance')
        common=[slot('weights',weights,'model-weights'),slot('tokenizer',TOKENIZER,'tokenizer')]
        value['input_contract'].update(requirements=common,requirement_bundles=[
            {'source_identity':'joy::ordinary','requirements':[slot('config',config,'artifact-input')],
             'additional_observed_readers':[{'reader':star,'role':'artifact-input'}]},
            {'source_identity':'joy::prism','requirements':[slot('config',config,'artifact-input'),slot('prism-metadata',prism,'artifact-input')],
             'additional_observed_readers':[{'reader':star,'role':'artifact-input'}]},
            {'source_identity':'joy::llava','requirements':[slot('llava-config',llava,'artifact-input')],
             'additional_observed_readers':[{'reader':config,'role':'artifact-input'},{'reader':star,'role':'artifact-input'}]},
            {'source_identity':'joy::star','requirements':[slot('star-config',star,'artifact-input')]}])
        def log(readers):
            return ''.join(event(item('/MOCK/'+str(i)),scope=value['fqn'],reader=reader,role=inputs.READERS[reader]) for i,reader in enumerate(readers))
        for extra in ([],[star]):
            result=inputs.assess_inputs(value,log([weights,TOKENIZER,config]+extra))
            self.assertEqual([x['source_identity'] for x in result['input_requirement_bundles']],['joy::ordinary'])
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value,log([weights,prism]))
        full=inputs.assess_inputs(value,log([weights,TOKENIZER,config,prism,star]))
        self.assertEqual([x['source_identity'] for x in full['input_requirement_bundles']],['joy::prism'])

    def test_five_mode_thinking_side_and_jin_support_preserve_prism_requirements(self):
        value=row();value.update(package='mlx-llm',fqn='thinking::qwen3_thinking_and_nothink')
        had='crates/llm/mlx-llm/src/prism.rs::from_dir::std::fs::read_to_string@5963'
        base=[slot('weights',DIRECTORY,'model-weights'),slot('config',CONFIG,'artifact-input'),slot('tokenizer',TOKENIZER,'tokenizer')]
        arms=[]
        for prism in (False,True):
            for template,reader,role in [('SIDE',SIDECAR,'artifact-input'),('JIN',JINJA,'tokenizer')]:
                requirements=copy.deepcopy(base)+([slot('hadamard',had,'artifact-input')] if prism else [])+[slot('template',reader,role)]
                support=[{'reader':SIDECAR,'role':'artifact-input'}] if template=='JIN' else []
                arms.append({'source_identity':('prism' if prism else 'ordinary')+'::'+template,
                             'requirements':requirements,'additional_observed_readers':support})
        arms.append({'source_identity':'direct::GGUF','requirements':[slot('gguf',GGUF,'tensor-input')]})
        value['input_contract'].update(requirements=[],requirement_bundles=arms)
        def log(readers):
            return ''.join(event(item('/MOCK/'+str(i)),scope=value['fqn'],reader=reader,role=inputs.READERS[reader]) for i,reader in enumerate(readers))
        for template,readers in [('SIDE',[SIDECAR]),('JIN',[SIDECAR,JINJA])]:
            text=log([DIRECTORY,CONFIG,TOKENIZER]+readers)
            ordinary=inputs.assess_inputs(value,text)
            self.assertEqual([x['source_identity'] for x in ordinary['input_requirement_bundles']],['ordinary::'+template])
            prism=inputs.assess_inputs(value,log([DIRECTORY,CONFIG,TOKENIZER,had]+readers))
            self.assertEqual([x['source_identity'] for x in prism['input_requirement_bundles']],['prism::'+template])
        with self.assertRaises(ValueError):
            inputs.assess_inputs(value,log([DIRECTORY,CONFIG,had,SIDECAR]))

    def test_mandatory_in_one_selector_support_only_in_other_stays_selector_local(self):
        value=row();value['input_contract'].update(requirements=[],requirement_bundles=[
            {'source_identity':'source::directory','requirements':[slot('weights',DIRECTORY,'model-weights'),slot('tokenizer',TOKENIZER,'tokenizer')]},
            {'source_identity':'source::gguf','requirements':[slot('weights',GGUF,'tensor-input')],
             'additional_observed_readers':[{'reader':TOKENIZER,'role':'tokenizer'}]}])
        inputs.assess_inputs(value,observation(DIRECTORY)+observation(TOKENIZER))
        result=inputs.assess_inputs(value,observation(GGUF)+observation(TOKENIZER))
        self.assertEqual([x['source_identity'] for x in result['input_requirement_bundles']],['source::gguf'])
        bad=copy.deepcopy(value)
        bad['input_contract']['requirement_bundles'][0]['additional_observed_readers']=[{'reader':TOKENIZER,'role':'tokenizer'}]
        with self.assertRaisesRegex(ValueError,'supporting.*mandatory'):
            inputs.validate_contract(bad,required=True)

    def test_source_proved_wan_file_directory_matrix_and_generated_equivalent(self):
        file_reader='mlx-gen::Weights::from_file';paths_reader='mlx-gen::Weights::from_paths'
        for generated in (False,True):
            value=row('mixed' if generated else 'external')
            common=[slot('T5-file',file_reader,'model-weights'),slot('VAE-file',file_reader,'model-weights')]
            if generated:
                common[0]['case_generated_write_read']=dict(WRITE_READ_PROOF)
                value['input_contract']['case_generated_inputs']=True
            value['input_contract'].update(requirements=common,requirement_bundles=[
                {'source_identity':'wan::FF','requirements':[{**slot('expert-files',file_reader,'model-weights'),'minimum_collections':2}]},
                {'source_identity':'wan::mixed','requirements':[slot('expert-file',file_reader,'model-weights'),slot('expert-dir',paths_reader,'model-weights')]},
                {'source_identity':'wan::DD','requirements':[{**slot('expert-dirs',paths_reader,'model-weights'),'minimum_collections':2}]}])
            for files,dirs,selected in [(4,0,'wan::FF'),(3,1,'wan::mixed'),(2,2,'wan::DD'),
                                        (2,1,None),(3,0,None),(2,0,None),(4,1,None),(3,2,None),(4,2,None)]:
                text=''.join(event(item('/MOCK/file-'+str(i)),reader=file_reader,role='model-weights') for i in range(files))
                text+=''.join(event(item('/MOCK/dir-'+str(i)),reader=paths_reader,role='model-weights') for i in range(dirs))
                with self.subTest(generated=generated,files=files,dirs=dirs):
                    if selected:
                        result=inputs.assess_inputs(value,text)
                        self.assertEqual([x['source_identity'] for x in result['input_requirement_bundles']],[selected])
                        self.assertFalse(result['terminal_model_evidence'])
                    else:
                        with self.assertRaises(ValueError):inputs.assess_inputs(value,text)

    def test_unrelated_observation_never_turns_absence_into_success(self):
        result = inputs.assess_inputs(self.case(), observation(GGUF))
        self.assertEqual(result['input_grade'], 'no-input-executed')
        self.assertFalse(result['checkpoint_byte_integrity_attested'])


if __name__ == '__main__':
    unittest.main()
