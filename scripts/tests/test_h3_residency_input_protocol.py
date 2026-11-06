"""Offline/SIMULATED protocol controls. No inference or model-file access.

The immutable C3 event extraction remains failed historical evidence. Added
validated-content packets below are SIMULATED metadata for the proposed protocol,
never a retroactive successful C3 workflow or newly executed model proof.
"""
import copy
import json
from pathlib import Path
import unittest

from scripts.ci import model_proof_inputs as grader

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'scripts/tests/fixtures/h3-residency-c3-input-events.json'
REGISTRY = ROOT / 'scripts/ci/model-inference-proofs.json'
C3 = 'dit::layers::residency_component::production_q4_turbo_workspace_and_qk_parity'
TE = 'text_encoder::encoder::residency_te::production_prompt_resident_window_first_difference'


def packets_text(packets):
    return ''.join('MODEL_INFERENCE_INPUT_USED ' + event['scope'] + ' ' +
                   json.dumps(event['packet'], separators=(',', ':')) + '\n' for event in packets)


def simulated_item(asset, manifest_sha):
    identity = {'display': asset['path'], 'os_encoded_hex': asset['path'].encode().hex()}
    return {'literal_path': identity, 'absolute_lexical_path': dict(identity),
            'os_encoding': 'rust-osstr-encoded', 'provenance': {
                'kind': 'validated-content', 'content_integrity': 'validated-existing-receipt',
                'sha256': asset['sha256'],
                'verifier': 'SIMULATED offline H3 fixture verifier; not runtime evidence',
                'source_binding': 'SIMULATED successful consumption after frozen manifest ' + manifest_sha}}


def simulated_event(scope, reader, role, assets, manifest_sha, weak=False):
    items = [simulated_item(asset, manifest_sha) for asset in assets]
    if weak:
        for item in items:
            item['provenance'] = {'kind': 'successful-reader-binding', 'content_integrity': 'not_attested'}
    return {'scope': scope, 'packet': {'schema_version': 1, 'reader': reader, 'role': role,
                                     'complete': True, 'items': items}}


class H3VerifiedInputProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding='utf-8'))
        rows = json.loads(REGISTRY.read_text(encoding='utf-8'))
        cls.rows = {row['fqn']: row for row in rows if row['fqn'] in (C3, TE)}

    def augmented_c3(self):
        events = copy.deepcopy(self.fixture['actualEvents'])
        assets = self.fixture['actualVerifiedAssets']
        groups = {'verified-shards': [a for a in assets if a['relative'].endswith('.safetensors')
                                     and a['relative'].startswith('q4/transformer/')],
                  'verified-config': [a for a in assets if a['relative'] == 'q4/transformer/config.json'],
                  'verified-turbo': [a for a in assets if a['relative'].startswith('minimax_h3_fl2v_turbo_8step')]}
        for requirement in self.rows[C3]['input_contract']['requirements']:
            if requirement['required_grade'] == 'validated-content':
                suffix = requirement['readers'][0].split('::')[-1]
                events.append(simulated_event(C3, requirement['readers'][0], requirement['role'],
                                             groups[suffix], self.fixture['actualFrozenManifestSha256']))
        return events

    def simulated_te(self):
        manifest = self.fixture['teManifestMetadataForSimulatedReceipts']
        assets = manifest['assets']
        shards = [a for a in assets if a['relative'].endswith('.safetensors')]
        configs = [a for a in assets if a['relative'] == 'q4/text_encoder/config.json']
        tokenizers = [a for a in assets if a['relative'].startswith('tokenizer/')]
        request = [{'path': manifest['requestPath'], 'sha256': manifest['requestSha256']}]
        groups = {'verified-shards': shards, 'verified-config': configs,
                  'verified-tokenizers': tokenizers, 'verified-request': request}
        events = []
        for requirement in self.rows[TE]['input_contract']['requirements']:
            reader = requirement['readers'][0]
            if requirement['required_grade'] == 'validated-content':
                events.append(simulated_event(TE, reader, requirement['role'],
                                             groups[reader.split('::')[-1]], '2bad494f27a89b6770f98fb0858ed1d89620cbc452716bcbd44a562dc92df113'))
            elif requirement['minimum_collections'] == 13:
                # Each actual mmap/header API returns ONE file per complete collection.
                for shard in shards:
                    events.append(simulated_event(TE, reader, requirement['role'], [shard], '', weak=True))
            else:
                selected = tokenizers if requirement['role'] == 'tokenizer' else configs
                events.append(simulated_event(TE, reader, requirement['role'], selected, '', weak=True))
        return events

    def test_actual_c3_remains_failed_under_its_original_content_contract(self):
        self.assertTrue(self.fixture['simulationOnly'])
        self.assertFalse(self.fixture['historicalWorkflowAcceptance'])
        self.assertEqual(self.fixture['sourceBodySha256'], '6ab6c0cbee355f99623c6b3e846a7c3bf82feb8aa40bcf77bb44539d8d2572c3')
        events = self.fixture['actualEvents']
        self.assertEqual(len(events), 31)
        self.assertEqual({item['provenance']['kind'] for event in events
                          for item in event['packet']['items']}, {'successful-reader-binding'})
        old = {**self.rows[C3], 'input_contract': self.fixture['historicalInputContract']}
        with self.assertRaisesRegex(ValueError, 'missing or partial'):
            grader.assess_inputs(old, packets_text(events))
        with self.assertRaises(ValueError):
            grader.assess_inputs(self.rows[C3], packets_text(events))

    def test_simulated_validated_augmentation_satisfies_real_grader_without_upgrading_actual_events(self):
        augmented = self.augmented_c3()
        self.assertEqual(augmented[:31], self.fixture['actualEvents'])
        self.assertEqual(len(augmented), 34)
        assessment = grader.assess_inputs(self.rows[C3], packets_text(augmented))
        self.assertEqual(len(assessment['required_input_slots']), 8)
        self.assertEqual(assessment['input_grade'], 'successful-reader-binding')
        # Actual weak API observations remain weak; the fixture adds its own honest strong slots.
        self.assertFalse(assessment['checkpoint_byte_integrity_attested'])

    def test_simulated_te_covers_single_file_collections_and_whole_verified_sets(self):
        assessment = grader.assess_inputs(self.rows[TE], packets_text(self.simulated_te()))
        self.assertEqual(len(assessment['required_input_slots']), 8)
        mmap = next(r for r in self.rows[TE]['input_contract']['requirements'] if r['slot'] == 'actual-te-shard-readers')
        self.assertEqual((mmap['minimum_collections'], mmap['minimum_items']), (13, 1))
        self.assertFalse(assessment['checkpoint_byte_integrity_attested'])

    def test_simulated_missing_underlying_reader_cannot_be_rescued_by_verified_content(self):
        for fqn, events in ((C3, self.augmented_c3()), (TE, self.simulated_te())):
            for reader in {r['readers'][0] for r in self.rows[fqn]['input_contract']['requirements']
                           if r['required_grade'] == 'reader-binding'}:
                with self.subTest(fqn=fqn, reader=reader), self.assertRaises(ValueError):
                    grader.assess_inputs(self.rows[fqn], packets_text([e for e in events if e['packet']['reader'] != reader]))

    def test_simulated_partial_wrong_digest_wrong_reader_or_incomplete_content_refuses(self):
        for fqn, base in ((C3, self.augmented_c3()), (TE, self.simulated_te())):
            for index, event in enumerate(base):
                if event['packet']['items'][0]['provenance']['kind'] != 'validated-content':
                    continue
                for mutation in ('partial', 'wrong-sha', 'wrong-reader', 'wrong-known-reader', 'incomplete'):
                    events = copy.deepcopy(base)
                    packet = events[index]['packet']
                    if mutation == 'partial':
                        packet['items'].pop()
                    elif mutation == 'wrong-sha':
                        packet['items'][0]['provenance']['sha256'] = 'f' * 64
                    elif mutation == 'wrong-reader':
                        packet['reader'] = 'unregistered::fake_content_reader'
                    elif mutation == 'wrong-known-reader':
                        catalog = json.loads((ROOT / 'scripts/ci/model-input-readers.json').read_text(encoding='utf-8'))
                        packet['reader'] = next(
                            reader for reader, capability in catalog['readers'].items()
                            if capability['role'] == packet['role']
                            and 'validated-content' not in capability['grades']
                        )
                    else:
                        packet['complete'] = False
                    with self.subTest(fqn=fqn, reader=event['packet']['reader'], mutation=mutation), self.assertRaises(ValueError):
                        grader.assess_inputs(self.rows[fqn], packets_text(events))

    def test_turbo_pin_and_fixture_only_reader_capabilities_remain_strong(self):
        turbo = next(r for r in self.rows[C3]['input_contract']['requirements'] if r['slot'] == 'actual-canonical-turbo')
        self.assertEqual(turbo['required_grade'], 'validated-content')
        self.assertEqual(turbo['expected_sha256'], ['e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f'])
        for row in self.rows.values():
            for requirement in row['input_contract']['requirements']:
                if requirement['required_grade'] == 'validated-content':
                    reader = requirement['readers'][0]
                    self.assertEqual(grader.READER_GRADES[reader], frozenset(('validated-content',)))
                    source = (ROOT / row['source']).read_text(encoding='utf-8')
                    self.assertIn('"' + reader + '"', source)
                    self.assertIn('InputIdentity::validated(', source)


if __name__ == '__main__':
    unittest.main()
