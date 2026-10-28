"""Scope-bound actual input receipts and explicit case-only/content claim separation.

Never reads model inputs, resolves a cache, hashes payloads, or infers identity from env/path metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

ROLES = frozenset(('model-weights', 'configuration', 'tokenizer', 'reference', 'tensor-input', 'artifact-input'))
KINDS = frozenset(('validated-content', 'same-read-buffer', 'successful-reader-binding'))
INTEGRITY = {'validated-content': 'validated-existing-receipt', 'same-read-buffer': 'same-buffer-sha256',
             'successful-reader-binding': 'not_attested'}
SHA = re.compile(r'[0-9a-f]{64}')
IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_.:/-]*')
# This source-controlled catalogue is generated from actual successful API hooks, not case counts.
_READER_CATALOGUE = json.loads(Path(__file__).with_name('model-input-readers.json').read_text(encoding='utf-8'))
if _READER_CATALOGUE.get('schema_version') != 1 or not isinstance(_READER_CATALOGUE.get('readers'), dict):
    raise ValueError('invalid actual-source input reader catalogue')
READERS = {reader: definition['role'] for reader, definition in _READER_CATALOGUE['readers'].items()}
READER_GRADES = {reader: frozenset(definition['grades']) for reader, definition in _READER_CATALOGUE['readers'].items()}
# A point-in-time verification is supporting context, not a successful model-consumption boundary.
VERIFICATION_ONLY = frozenset(_READER_CATALOGUE.get('supporting_not_consumption', []))
# A source producer that emits one ordered list of records (for example the SDXL dual-CLIP
# tokenizer pair) may legitimately repeat a lexical path when both source arguments alias.
# The flagged set, each producer's exact record ARITY, and the authorizing source-fact hash
# are all bound to the retained frozen alias packet: any other flagged catalogue entry,
# arity, or provenance hash rejects the whole catalogue before any receipt is read.
_ORDERED_RECORD_SOURCE_FACTS_SHA256 = '888da778962d6902bcf54ab53b83436e5677abc22668902d3421a23b67506f44'
_ORDERED_RECORD_FLAGGED_ARITY = {
    'candle-gen-sdxl::pipeline::SdxlTokenizers::load/tokenizers': ('tokenizer', 2),
    'candle-gen-sdxl::training::DualClip::load/tokenizers': ('tokenizer', 2)}
if _READER_CATALOGUE.get('ordered_record_source_facts_sha256') != _ORDERED_RECORD_SOURCE_FACTS_SHA256:
    raise ValueError('ordered-record producers are bound to the retained frozen source facts')
ORDERED_RECORD_ARITY = {}
for _reader, _definition in _READER_CATALOGUE['readers'].items():
    if 'ordered_records' in _definition:
        if _definition['ordered_records'] is not True:
            raise ValueError('ordered_records is an exact frozen-source flag')
        _expected = _ORDERED_RECORD_FLAGGED_ARITY.get(_reader)
        if (_expected is None or _definition.get('ordered_records_arity') != _expected[1]
                or _definition['role'] != _expected[0]):
            raise ValueError('ordered-record producers are bound to the retained frozen source facts')
        ORDERED_RECORD_ARITY[_reader] = _expected[1]
if set(ORDERED_RECORD_ARITY) != set(_ORDERED_RECORD_FLAGGED_ARITY):
    raise ValueError('ordered-record producers are bound to the retained frozen source facts')
ORDERED_RECORD_READERS = frozenset(ORDERED_RECORD_ARITY)


CLAIM_FIELDS = ('acceptance_scope', 'input_grade', 'input_grades', 'input_claim_limits',
                'required_input_content_attested', 'checkpoint_byte_integrity_attested',
                'verified_named_revision', 'terminal_model_evidence')


def receipt_claims(cases: list[dict], accepted: bool) -> dict:
    grades = {case.get('input_grade', 'unresolved') for case in cases}
    no_input = 'no-input-executed' in grades
    accepted = accepted and not no_input
    weak = bool(grades & {'successful-reader-binding', 'source-defined', 'unresolved', 'no-input-executed'})
    limits = sorted({limit for case in cases for limit in case.get('input_claim_limits', [])})
    if not accepted:
        limits = sorted(set(limits) | {'incomplete/unaccepted evidence; no model or checkpoint PASS'})
    grade = ('no-input-executed' if no_input else
             'unresolved' if not accepted or not grades or 'unresolved' in grades else
             'successful-reader-binding' if 'successful-reader-binding' in grades else
             'source-defined' if 'source-defined' in grades else
             'same-read-buffer' if 'same-read-buffer' in grades else 'validated-content')
    return {'acceptance_scope': ('unaccepted-incomplete' if not accepted else
                                'source-bound-case-execution' if weak else 'content-bound-case-execution'),
            'input_grade': grade, 'input_grades': sorted(grades), 'input_claim_limits': limits,
            'required_input_content_attested': bool(accepted and cases and all(case.get('required_input_content_attested') is True for case in cases)),
            'checkpoint_byte_integrity_attested': bool(accepted and cases and all(case.get('checkpoint_byte_integrity_attested') is True for case in cases)),
            'verified_named_revision': False, 'terminal_model_evidence': False}


def _flatten_bundle(contract: dict, bundle: dict) -> dict:
    flattened = {key: value for key, value in contract.items() if key != 'requirement_bundles'}
    flattened['requirements'] = contract['requirements'] + bundle['requirements']
    # The flag authorizes anchored slots across the contract UNION, not generated work in
    # every source alternative. An arm with no generated slots has only inert metadata.
    if (flattened.get('case_generated_inputs') is True
            and not any('case_generated_write_read' in item for item in flattened['requirements'])):
        flattened.pop('case_generated_inputs')
    return flattened


def validate_contract(row: dict, *, required: bool = False) -> dict | None:
    contract = row.get('input_contract')
    if contract is None:
        if required:
            raise ValueError('actual input contract is unresolved; no source-only external runtime acceptance')
        return None
    if (not isinstance(contract, dict) or not {'fixture_kind', 'source_identity', 'requirements'} <= set(contract)
            or not set(contract) <= {'fixture_kind', 'source_identity', 'requirements', 'reader_roles', 'source_defined_default', 'requirement_bundles', 'case_generated_inputs', 'conditional_only'}):
        raise ValueError('input contract must declare source fixture kind, actual required slots and reader roles')
    if not isinstance(contract['fixture_kind'], str) or contract['fixture_kind'] not in ('source-defined', 'external', 'mixed', 'unresolved'):
        raise ValueError('unknown input fixture kind')
    if not isinstance(contract['source_identity'], str) or not contract['source_identity']:
        raise ValueError('input contract needs an explicit source identity')
    if 'source_defined_default' in contract and (type(contract['source_defined_default']) is not bool
            or contract['fixture_kind'] not in ('mixed', 'external')):
        raise ValueError('source-defined fallback must be an explicit actual optional-slot branch contract')
    readers = dict(READERS)
    extra = contract.get('reader_roles', {})
    if not isinstance(extra, dict):
        raise ValueError('input reader roles must be exact source-bound IDs')
    for reader, role in extra.items():
        if (not isinstance(reader, str) or not isinstance(role, str) or role not in ROLES
                or reader in VERIFICATION_ONLY or reader not in READERS or READERS[reader] != role):
            raise ValueError('invalid or role-changing input reader declaration')
        readers[reader] = role
    requirements = contract['requirements']
    if not isinstance(requirements, list):
        raise ValueError('input requirements must be a list of source-bound slots')
    # Same-case generated files remain observable obligations even beside external inputs.
    # The contract flag authorizes only slots with their own write/read anchors; it never
    # changes the kind or excuses any non-generated external input.
    if 'case_generated_inputs' in contract and (type(contract['case_generated_inputs']) is not bool
            or contract['fixture_kind'] not in ('source-defined', 'mixed', 'external')):
        raise ValueError('case-generated inputs require an explicit resolved fixture declaration')
    if contract['fixture_kind'] == 'source-defined' and requirements and not contract.get('case_generated_inputs'):
        raise ValueError('source-defined-only fixture cannot disguise required external slots')
    bundles = contract.get('requirement_bundles')
    if 'requirement_bundles' in contract:
        if (contract['fixture_kind'] not in ('external', 'mixed') or not isinstance(bundles, list)
                or len(bundles) < 2):
            raise ValueError('source-mode alternatives require complete source-bound external bundles')
        identities = set()
        for bundle in bundles:
            if (not isinstance(bundle, dict) or not {'source_identity', 'requirements'} <= set(bundle)
                    or not set(bundle) <= {'source_identity', 'requirements', 'additional_observed_readers'}
                    or not isinstance(bundle['source_identity'], str) or not bundle['source_identity']
                    or bundle['source_identity'] in identities
                    or not isinstance(bundle['requirements'], list) or not bundle['requirements']):
                raise ValueError('each source-mode bundle needs a unique source identity and nonempty AND requirements')
            additional = bundle.get('additional_observed_readers', [])
            if not isinstance(additional, list):
                raise ValueError('additional observations must be exact source-declared reader/role pairs')
            pairs = set()
            for observed in additional:
                if (not isinstance(observed, dict) or set(observed) != {'reader', 'role'}
                        or not isinstance(observed['reader'], str) or not isinstance(observed['role'], str)
                        or observed['reader'] in VERIFICATION_ONLY
                        or readers.get(observed['reader']) != observed['role']
                        or (observed['reader'], observed['role']) in pairs):
                    raise ValueError('additional observations cannot change reader roles or authorize unknown inputs')
                pairs.add((observed['reader'], observed['role']))
            identities.add(bundle['source_identity'])
            flattened = _flatten_bundle(contract, bundle)
            validate_contract({**row, 'input_contract': flattened}, required=required)
            mandatory_pairs = {(reader, item['role']) for item in flattened['requirements']
                               for reader in item['readers']}
            if pairs & mandatory_pairs:
                raise ValueError('supporting pairs cannot also satisfy mandatory work in the same selector')
    if contract['fixture_kind'] in ('external', 'mixed') and not requirements and not bundles:
        raise ValueError('external/mixed fixture requires actual slot completeness, not any one receipt')
    slots = set()
    for requirement in requirements:
        fields = {'slot', 'role', 'readers', 'minimum_collections', 'minimum_items', 'required_grade', 'scope'}
        if (not isinstance(requirement, dict) or not fields <= set(requirement)
                or not set(requirement) <= fields | {'expected_sha256', 'when', 'case_generated_write_read'}):
            raise ValueError('input slot has missing/unknown completeness fields')
        if 'when' in requirement:
            # external-selected: an optional override with a positive source-defined default.
            # external-present: an optional external input with NO source-defined default —
            # the obligation exists exactly when the input is actually observed; absence
            # imposes nothing and never fabricates a receipt or an env/path heuristic.
            if requirement['when'] == 'external-present':
                if (contract['fixture_kind'] not in ('mixed', 'external')
                        or requirement['scope'] == 'each-child'):
                    raise ValueError('external-present is an exact optional external branch with case scope')
            elif (requirement['when'] != 'external-selected'
                    or contract['fixture_kind'] not in ('mixed', 'external') or contract.get('source_defined_default') is not True):
                raise ValueError('conditional input slot requires exact consumed external/source-defined branch semantics')
        proof = requirement.get('case_generated_write_read')
        if contract.get('case_generated_inputs') is True and (proof is not None or contract['fixture_kind'] == 'source-defined'):
            # Every flagged requirement names the exact same-case writer anchor and its source
            # file hash itself; no family-level inference, filename, or absence proof counts.
            if (not isinstance(proof, dict) or set(proof) != {'case_source', 'case_source_sha256', 'writer'}
                    or not isinstance(proof['case_source'], str) or not proof['case_source']
                    or not isinstance(proof['case_source_sha256'], str) or not SHA.fullmatch(proof['case_source_sha256'])
                    or proof['case_source_sha256'] == '0' * 64
                    or not isinstance(proof['writer'], str) or not proof['writer']):
                raise ValueError('each case-generated slot needs its exact same-case write-then-read anchor')
        elif 'case_generated_write_read' in requirement:
            raise ValueError('write-then-read anchors belong only to declared case-generated contracts')
        slot = requirement['slot']
        if not isinstance(slot, str) or not IDENTIFIER.fullmatch(slot) or slot in slots:
            raise ValueError('input slots must have exact unique source identities')
        slots.add(slot)
        allowed = requirement['readers']
        if (not isinstance(requirement['role'], str) or requirement['role'] not in ROLES or not isinstance(allowed, list) or not allowed
                or any(not isinstance(reader, str) or readers.get(reader) != requirement['role'] for reader in allowed)
                or len(set(allowed)) != len(allowed)):
            raise ValueError('required input role/reader is not an exact successful source boundary')
        if any(type(requirement[key]) is not int or requirement[key] < 1 for key in ('minimum_collections', 'minimum_items')):
            raise ValueError('required input completeness must be strictly positive')
        if requirement['required_grade'] not in ('reader-binding', 'content', 'validated-content'):
            raise ValueError('unknown required input provenance grade')
        if requirement['scope'] not in ('case', 'each-child'):
            raise ValueError('input slots must bind exact case/child execution scope')
        if requirement['scope'] == 'each-child' and row['execution_role'] != 'indexed_children':
            raise ValueError('per-child inputs require an existing declared indexed-child topology')
        pins = requirement.get('expected_sha256')
        if pins is not None and (not isinstance(pins, list) or not pins or any(not isinstance(value, str) or not SHA.fullmatch(value) for value in pins)):
            raise ValueError('existing content pins must be exact SHA256 values')
        if pins is not None and requirement['required_grade'] == 'reader-binding':
            raise ValueError('a pinned input slot cannot downgrade to weak reader binding')
        # The frozen source fact fixes how many ordered records each flagged producer emits;
        # a mapper-authored minimum_items can never authorize a different multiplicity.
        arities = {ORDERED_RECORD_ARITY[reader] for reader in allowed if reader in ORDERED_RECORD_ARITY}
        if arities:
            if len(arities) != 1 or len(allowed) != 1 or requirement['minimum_items'] != arities.pop():
                raise ValueError('an ordered-record slot must bind exactly its source producer arity')
    all_requirements = requirements + [item for bundle in bundles or [] for item in bundle['requirements']]
    present_pairs = {(reader, item['role'], item['scope']) for item in all_requirements
                     if item.get('when') == 'external-present' for reader in item['readers']}
    default_pairs = {(reader, item['role'], item['scope']) for item in all_requirements
                     if item.get('when') == 'external-selected' for reader in item['readers']}
    if present_pairs & default_pairs:
        raise ValueError('optional presence and source-default slots must have disjoint reader/role/scope pairs')
    if present_pairs and contract.get('source_defined_default') is True and not default_pairs:
        raise ValueError('a source-defined default must bind its own distinct external-selected slot')
    if contract.get('case_generated_inputs') is True and contract['fixture_kind'] in ('external', 'mixed'):
        complete_sets = ([requirements + bundle['requirements'] for bundle in bundles]
                         if bundles else [requirements])
        if (not any('case_generated_write_read' in item for items in complete_sets for item in items)
                or any(not any('case_generated_write_read' not in item for item in items)
                       for items in complete_sets)):
            raise ValueError('external/mixed generated slots must retain non-generated input requirements')
    if 'conditional_only' in contract:
        complete_sets = ([requirements + bundle['requirements'] for bundle in bundles]
                         if bundles else [requirements])
        if (contract['conditional_only'] is not True or contract['fixture_kind'] != 'external'
                or 'source_defined_default' in contract
                or any(not items or any(item.get('when') != 'external-present' for item in items)
                       for items in complete_sets)):
            raise ValueError('conditional_only requires external-only, wholly optional actual inputs without a default')
    if contract['fixture_kind'] == 'external' and contract.get('conditional_only') is not True:
        # Optional branches (external-selected defaults, external-present goldens) can never
        # turn an external case into a no-input acceptance: every complete flattened set
        # retains at least one unconditional external requirement.
        def _retains_unconditional(items: list) -> bool:
            return any('when' not in requirement and 'case_generated_write_read' not in requirement
                       for requirement in items)
        if bundles:
            if not all(_retains_unconditional(requirements + bundle['requirements']) for bundle in bundles):
                raise ValueError('every external source-mode bundle must retain unconditional external input work')
        elif requirements and not _retains_unconditional(requirements):
            raise ValueError('every external source-mode bundle must retain unconditional external input work')
    if required and contract['fixture_kind'] == 'unresolved':
        raise ValueError('actual external input slots/readers remain unresolved; no runtime acceptance')
    return contract


def path_identity(value: object) -> str:
    if (not isinstance(value, dict) or set(value) != {'display', 'os_encoded_hex'}
            or not isinstance(value['display'], str) or not isinstance(value['os_encoded_hex'], str)
            or not re.fullmatch(r'(?:[0-9a-f]{2})+', value['os_encoded_hex'])):
        raise ValueError('input path must retain actual display and lossless OS bytes, never a metadata hash')
    return value['os_encoded_hex']


def validate_item(item: object) -> tuple[str, str, str | None]:
    if not isinstance(item, dict) or 'provenance' not in item:
        raise ValueError('input item lacks actual provenance')
    provenance = item['provenance']
    if (not isinstance(provenance, dict) or not isinstance(provenance.get('kind'), str)
            or provenance['kind'] not in KINDS):
        raise ValueError('unknown input provenance kind')
    kind = provenance['kind']
    allowed_fields = {'kind', 'content_integrity'} | ({'claim_limit'} if kind == 'successful-reader-binding' else {'sha256'})
    if kind == 'validated-content':
        allowed_fields |= {'verifier', 'source_binding'}
    if not set(provenance) <= allowed_fields:
        raise ValueError('input provenance has unknown/hash-like fields outside its actual grade')
    if 'claim_limit' in provenance and (not isinstance(provenance['claim_limit'], str) or not provenance['claim_limit']):
        raise ValueError('weak input provenance claim limit must remain explicit text')
    if provenance.get('content_integrity') != INTEGRITY[kind]:
        raise ValueError('input provenance cannot upgrade or disguise its integrity limit')
    if kind == 'successful-reader-binding':
        if 'sha256' in provenance:
            raise ValueError('path/mtime/catalogue cannot masquerade as consumed content SHA256')
        digest = None
    else:
        digest = provenance.get('sha256')
        if not isinstance(digest, str) or not SHA.fullmatch(digest) or digest == '0' * 64:
            raise ValueError('strong input provenance requires an actual finalized content digest')
        if kind == 'validated-content' and any(not isinstance(provenance.get(key), str) or not provenance[key] for key in ('verifier', 'source_binding')):
            raise ValueError('validated content must retain actual verifier/consumption-seal binding')
    if item.get('origin') == 'memory-buffer':
        if kind != 'same-read-buffer' or set(item) != {'origin', 'provenance'}:
            raise ValueError('pathless inputs must be actual same-read owned buffers, not invented files')
        identity = 'memory-buffer:' + digest
    else:
        required = {'literal_path', 'absolute_lexical_path', 'os_encoding', 'provenance'}
        if not required <= set(item) or not set(item) <= required | {'source_origin'} or item['os_encoding'] != 'rust-osstr-encoded':
            raise ValueError('filesystem input must retain exact literal/absolute lexical identities')
        if 'source_origin' in item and item['source_origin'] not in ('source-defined-default', 'external-selected'):
            raise ValueError('input origin must be actual source selection, never path/env guessing')
        path_identity(item['literal_path'])
        identity = path_identity(item['absolute_lexical_path'])
    return identity, kind, digest


def _receipt_satisfies(requirement: dict, receipt: dict, row: dict) -> bool:
    scopes = ([row['fqn'] + '#child=' + child for child in row['expected_child_ids']]
              if requirement['scope'] == 'each-child' else [row['fqn']])
    if (receipt['scope'] not in scopes or receipt['reader'] not in requirement['readers']
            or receipt['role'] != requirement['role']):
        return False
    selected = [item for item in receipt['items']
                if (item.get('source_origin') == 'external-selected' if requirement.get('when') == 'external-selected'
                    else item.get('source_origin') != 'source-defined-default')]
    if len(selected) < requirement['minimum_items']:
        return False
    values = [validate_item(item) for item in selected]
    if requirement['required_grade'] != 'reader-binding' and any(kind == 'successful-reader-binding' for _, kind, _ in values):
        return False
    if requirement['required_grade'] == 'validated-content' and any(kind != 'validated-content' for _, kind, _ in values):
        return False
    return ('expected_sha256' not in requirement
            or sorted(value[2] for value in values) == sorted(requirement['expected_sha256']))


def _requirement_implies(left: dict, right: dict) -> bool:
    # Observable implication only: no filename, model label, source-path or env inference.
    grades = {'reader-binding': 0, 'content': 1, 'validated-content': 2}
    return (left['role'] == right['role'] and left['scope'] == right['scope']
            and left.get('when') == right.get('when')
            and set(left['readers']) <= set(right['readers'])
            and left['minimum_collections'] >= right['minimum_collections']
            and left['minimum_items'] >= right['minimum_items']
            and grades[left['required_grade']] >= grades[right['required_grade']]
            and (not right.get('expected_sha256')
                 or left.get('expected_sha256') == right.get('expected_sha256')))


def _pair_collection_bounds(requirements: list[dict]) -> dict:
    bounds = {}
    for item in requirements:
        for reader in item['readers']:
            key = (reader, item['role'], item['scope'], item.get('when'))
            bounds[key] = bounds.get(key, 0) + item['minimum_collections']
    return bounds


def _bundle_requirement_classes(contract: dict, observed_pairs: set[tuple[str, str]] | None = None) -> list[dict]:
    bundles = contract['requirement_bundles']
    all_sets = [contract['requirements'] + bundle['requirements'] for bundle in bundles]
    mandatory_pairs = [{(reader, item['role']) for item in items for reader in item['readers']} for items in all_sets]
    supporting_pairs = [{(item['reader'], item['role']) for item in bundle.get('additional_observed_readers', [])}
                        for bundle in bundles]
    all_bounds = [_pair_collection_bounds(items) for items in all_sets]
    pairs = {pair for bounds in all_bounds for pair in bounds}
    floors = {pair: min(bounds.get(pair, 0) for bounds in all_bounds) for pair in pairs}
    result = []
    for current_index, (requirements, bounds) in enumerate(zip(all_sets, all_bounds)):
        shared, specific, remaining, unit_bounds = [], [], {}, {}
        for item in requirements:
            # Count independent collection units, not just the largest individual slot.
            # Four common file slots imply a floor of four, even when every slot says one.
            semantic = json.dumps({key: value for key, value in item.items()
                                   if key not in ('slot', 'minimum_collections', 'case_generated_write_read')}, sort_keys=True)
            unit = {**item, 'minimum_collections': 1}
            item_pairs = {(reader, item['role']) for reader in item['readers']}
            active_pairs = item_pairs & observed_pairs if observed_pairs is not None else item_pairs
            active_pairs = active_pairs or item_pairs
            # A selector retained ONLY through a supporting probe for this pair is not a
            # counterexample to required sharing. It must neither activate nor poison the
            # ordinary/Prism family (or SIDE family) that actually REQUIRES the pair.
            relevant = [index for index in range(len(all_sets))
                        if not (active_pairs <= supporting_pairs[index]
                                and active_pairs.isdisjoint(mandatory_pairs[index]))]
            capacity = min(sum(other['minimum_collections'] for other in all_sets[index]
                               if _requirement_implies({**other, 'minimum_collections': 1}, unit))
                           for index in relevant)
            arm_units = sum(other['minimum_collections'] for other in requirements
                            if _requirement_implies({**other, 'minimum_collections': 1}, unit))
            item_floors = {pair: min(all_bounds[index].get(pair, 0) for index in relevant)
                           for pair in floors}
            other_max = max((sum(other['minimum_collections'] for other in all_sets[index]
                                 if _requirement_implies({**other, 'minimum_collections': 1}, unit))
                             for index in relevant if index != current_index), default=0)
            unit_bounds[item['slot']] = {'shared': capacity, 'arm': arm_units,
                                        'other_max': other_max, 'pair_floors': item_floors}
            if semantic not in remaining:
                remaining[semantic] = capacity
            if remaining[semantic] >= item['minimum_collections']:
                shared.append(item)
            else:
                specific.append(item)
            remaining[semantic] = max(0, remaining[semantic] - item['minimum_collections'])
        result.append({'shared': shared, 'specific': specific,
                       'collection_floors': floors, 'arm_collections': bounds,
                       'unit_bounds': unit_bounds})
    return result


def _requirements_imply(left: list[dict], right: list[dict]) -> bool:
    supplied = [{**item, 'minimum_collections': 1} for item in left for _ in range(item['minimum_collections'])]
    needed = [{**item, 'minimum_collections': 1} for item in right for _ in range(item['minimum_collections'])]
    owners = {}
    def assign(index: int, seen: set[int]) -> bool:
        for other, value in enumerate(supplied):
            if other in seen or not _requirement_implies(value, needed[index]):
                continue
            seen.add(other)
            if other not in owners or assign(owners[other], seen):
                owners[other] = index
                return True
        return False
    return all(assign(index, set()) for index in range(len(needed)))


def _specific_sets_nested(left: list[dict], right: list[dict]) -> bool:
    return _requirements_imply(left, right) or _requirements_imply(right, left)


def _specific_observed(requirement: dict, receipts: list[dict], row: dict,
                       shared_requirements: list[dict], ignored_pairs: set[tuple[str, str]],
                       collection_floors: dict, unit_bounds: dict) -> bool:
    scopes = ([row['fqn'] + '#child=' + child for child in row['expected_child_ids']]
              if requirement['scope'] == 'each-child' else [row['fqn']])
    for scope in scopes:
        observed = [receipt for receipt in receipts
                    if receipt['scope'] == scope and receipt['reader'] in requirement['readers']
                    and receipt['role'] == requirement['role']
                    and (receipt['reader'], receipt['role']) not in ignored_pairs
                    and (requirement.get('when') != 'external-selected'
                         or any(item.get('source_origin') == 'external-selected' for item in receipt['items']))]
        if not observed:
            continue
        pairs = [(receipt['reader'], receipt['role'], requirement['scope'], requirement.get('when'))
                 for receipt in observed]
        effective_floors = unit_bounds[requirement['slot']]['pair_floors']
        if any(effective_floors.get(pair, 0) == 0 for pair in pairs):
            # Any positive mode-only API observation activates, even below grade/items.
            return True
        shared = [item for item in shared_requirements
                  if item['role'] == requirement['role'] and item['scope'] == requirement['scope']
                  and item.get('when') == requirement.get('when')
                  and set(item['readers']) & set(requirement['readers'])]
        # Aggregate compatible collection units with THIS role/scope/grade/pin predicate.
        # Four one-collection slots mean FOUR, while two distinct original pins do not
        # activate each other merely because they happen to use the same reader API.
        bound = unit_bounds[requirement['slot']]
        matching = [receipt for receipt in observed if _receipt_satisfies(requirement, receipt, row)]
        # Excess generic units distinguish an incomplete refinement only beyond EVERY
        # other compatible source plan's capacity. FF4 is not evidence of mixed F3+D1.
        if bound['arm'] > bound['other_max'] and len(matching) > bound['other_max']:
            return True
        if not shared:
            # Same API/cardinality but different original pins/grades is distinguished
            # only by actual matching provenance, not the mere shared API observation.
            return any(_receipt_satisfies(requirement, receipt, row) for receipt in observed)
        items = max(item['minimum_items'] for item in shared)
        if requirement['minimum_items'] > items and any(len(receipt['items']) > items for receipt in observed):
            return True
        grades = {'reader-binding': 0, 'content': 1, 'validated-content': 2}
        if grades[requirement['required_grade']] > max(grades[item['required_grade']] for item in shared):
            if any(_receipt_satisfies(requirement, receipt, row) for receipt in observed):
                return True
        if requirement.get('expected_sha256') and not all(item.get('expected_sha256') == requirement['expected_sha256'] for item in shared):
            if any(_receipt_satisfies(requirement, receipt, row) for receipt in observed):
                return True
    return False


def assess_inputs(row: dict, text: str) -> dict:
    contract = validate_contract(row, required=True)
    bundles = contract.get('requirement_bundles')
    if not bundles:
        return _assess_inputs(row, text)
    # OR of complete source-defined AND bundles, never a reader/role Cartesian product or
    # an operator-selected mode. Every attempt parses and grades the ENTIRE observation log.
    declared = contract['requirements'] + [requirement for bundle in bundles for requirement in bundle['requirements']]
    mode_pairs = {(reader, requirement['role']) for requirement in declared for reader in requirement['readers']}
    supporting_pairs = {(observed['reader'], observed['role']) for bundle in bundles
                        for observed in bundle.get('additional_observed_readers', [])}
    mode_pairs.update(supporting_pairs)
    observed = _observed_receipts(row, text, {**READERS, **contract.get('reader_roles', {})})
    observed_pairs = {(receipt['reader'], receipt['role']) for receipt in observed}
    classes = _bundle_requirement_classes(contract, observed_pairs)
    potential = []
    for index, bundle in enumerate(bundles):
        explained = {(reader, item['role']) for item in contract['requirements'] + bundle['requirements']
                     for reader in item['readers']}
        explained.update((item['reader'], item['role']) for item in bundle.get('additional_observed_readers', []))
        if not ((observed_pairs & mode_pairs) - explained):
            potential.append(index)
    # Nested source families share requirements with a subset of the case's arms (ordinary
    # Llama and Prism, unlike JoyCaption). Positive compatibility narrows that family first;
    # its shared tokenizer/config calls must not activate a different nested arm.
    family_classes = (_bundle_requirement_classes({**contract, 'requirement_bundles': [bundles[i] for i in potential]}, observed_pairs)
                      if potential else [])
    activation_classes = dict(zip(potential, family_classes))
    fully_satisfied = set()
    complete_assessments = []
    accepted = []
    for bundle in bundles:
        flattened = _flatten_bundle(contract, bundle)
        try:
            assessment = _assess_inputs({**row, 'input_contract': flattened}, text,
                                        observed_requirements=declared, supporting_pairs=supporting_pairs)
        except ValueError:
            continue
        fully_satisfied.add(bundle['source_identity'])
        complete_assessments.append((bundle, assessment))
        explained = {(reader, requirement['role']) for requirement in flattened['requirements']
                     for reader in requirement['readers']}
        explained.update((observed['reader'], observed['role'])
                         for observed in bundle.get('additional_observed_readers', []))
        observed_pairs = {(receipt['reader'], receipt['role']) for receipt in assessment['actual_inputs']}
        if (observed_pairs & mode_pairs) - explained:
            # Positive specific-mode evidence (for example Hadamard) cannot be discarded to
            # accept a smaller ordinary-directory subset. Source-declared optional attempted
            # pairs explain whitespace-sidecar + Jinja fallback without becoming required work.
            continue
        accepted.append((bundle, assessment))
    # Supporting observations are typed per compatible selector. A declined probe or an
    # empty sidecar attempt explained by a complete fallback never activates another arm.
    ignored = {(item['reader'], item['role']) for bundle, _ in accepted
               for item in bundle.get('additional_observed_readers', [])}
    shared = [requirement for classification in (classes if len(potential) == 1 else family_classes)
              for requirement in classification['shared']]
    classifications = []
    for index, (bundle, classification) in enumerate(zip(bundles, classes)):
        family = activation_classes.get(index)
        activation_requirements = (classification['specific'] if len(potential) == 1
                                   else family['specific'] if family else [])
        activation_profile = classification if len(potential) == 1 else family
        activated = bool(family and any(_specific_observed(requirement, observed, row, shared, ignored,
                                                          activation_profile['collection_floors'],
                                                          activation_profile['unit_bounds'])
                                        for requirement in activation_requirements))
        full = bundle['source_identity'] in fully_satisfied
        specific = family['specific'] if family is not None else classification['specific']
        full_own_observed = full and any(_receipt_satisfies(requirement, receipt, row)
                                        for requirement in specific for receipt in observed
                                        if (receipt['reader'], receipt['role']) not in ignored)
        if activated and not full:
            raise ValueError('activated source-mode arm is partial; shared or ordinary work cannot swallow it')
        classifications.append({'source_identity': bundle['source_identity'],
                                'shared_slots': [item['slot'] for item in classification['shared']],
                                'mode_specific_slots': [item['slot'] for item in classification['specific']],
                                'positive_observation_compatible': family is not None,
                                'family_shared_slots': [item['slot'] for item in family['shared']] if family else [],
                                'family_specific_slots': [item['slot'] for item in family['specific']] if family else [],
                                'activated': activated, 'fully_satisfied': full,
                                'full_own_required_observation': full_own_observed})
    eligible = {item['source_identity'] for item in classifications
                if item['positive_observation_compatible'] and (item['activated'] or not item['family_specific_slots']
                                                                or item['full_own_required_observation'])}
    selected = [bundle for bundle in bundles
                if bundle['source_identity'] in eligible and bundle['source_identity'] in fully_satisfied]
    # An extra positive API cannot erase an otherwise RAW-FULL nonnested plan (FF4
    # beside mixed F3+D1). Dismiss it only when a complete compatible selector explains
    # every unit through exact mandatory implication plus its explicitly typed probes.
    selected_ids = {bundle['source_identity'] for bundle in selected}
    by_class = {item['source_identity']: item for item in classifications}
    for outside, _ in complete_assessments:
        name = outside['source_identity']
        if name in selected_ids or not by_class[name]['full_own_required_observation']:
            continue
        required = contract['requirements'] + outside['requirements']
        explained_as_probe = False
        for selector in selected:
            probes = {(item['reader'], item['role']) for item in selector.get('additional_observed_readers', [])}
            unaccounted = [item for item in required
                           if not {(reader, item['role']) for reader in item['readers']} <= probes]
            supplied = contract['requirements'] + selector['requirements']
            if _requirements_imply(supplied, unaccounted):
                explained_as_probe = True
                break
        if explained_as_probe:
            continue
        for selector in selected:
            other = contract['requirements'] + selector['requirements']
            if not _requirements_imply(required, other) and not _requirements_imply(other, required):
                raise ValueError('raw-full nonnested source-mode plan cannot be erased by extra observations')
    if not potential and len(fully_satisfied) > 1:
        raise ValueError('multiple disjoint fully satisfied source-mode arms are ambiguous')
    by_identity = {bundle['source_identity']: activation_classes.get(index, classification)
                   for index, (bundle, classification) in enumerate(zip(bundles, classes))}
    for index, left in enumerate(selected):
        for right in selected[index + 1:]:
            a = by_identity[left['source_identity']]['specific']
            b = by_identity[right['source_identity']]['specific']
            if not a and not b:
                raise ValueError('identical mode observables require a source producer witness, not guessed selection')
            if not _specific_sets_nested(a, b):
                raise ValueError('multiple disjoint fully satisfied source-mode arms are ambiguous')
    accepted = [(bundle, assessment) for bundle, assessment in accepted
                if bundle['source_identity'] in eligible]
    if not accepted:
        raise ValueError('no complete source-mode input requirement bundle was observed')
    result = dict(accepted[0][1])
    result['input_bundle_requirement_classes'] = classifications
    for receipt in result['actual_inputs']:
        applicable = [requirement for requirement in declared
                      if receipt['reader'] in requirement['readers'] and receipt['role'] == requirement['role']]
        source_default = (applicable and all(requirement.get('when') == 'external-selected' for requirement in applicable)
                          and all(item.get('source_origin') == 'source-defined-default' for item in receipt['items']))
        if applicable and not source_default and not any(_receipt_satisfies(requirement, receipt, row) for requirement in applicable):
            raise ValueError('an observed source-mode input violates its declared scope, completeness, grade or pin')
    # These are SATISFIED observation bundles, not a guessed branch predicate. A successful
    # empty/whitespace sidecar read followed by Jinja fallback can leave both read events.
    result['input_requirement_bundles'] = [
        {'source_identity': bundle['source_identity'], 'required_input_slots': assessment['required_input_slots']}
        for bundle, assessment in accepted]
    for key in ('required_input_slots', 'input_source_branches'):
        combined = []
        for _, assessment in accepted:
            for value in assessment[key]:
                if value not in combined:
                    combined.append(value)
        result[key] = combined
    for key in ('required_input_content_attested', 'checkpoint_byte_integrity_attested'):
        result[key] = all(assessment[key] for _, assessment in accepted)
    return result


def _observed_receipts(row: dict, text: str, readers: dict) -> list[dict]:
    root = row['fqn']
    scopes = {root}
    if row['execution_role'] == 'indexed_children':
        scopes.update(root + '#child=' + child for child in row['expected_child_ids'])
    receipts = []
    seen = set()
    for line in text.splitlines():
        if not line.startswith('MODEL_INFERENCE_INPUT_USED '):
            continue
        fields = line.split(' ', 2)
        if len(fields) != 3 or fields[1] not in scopes:
            raise ValueError('input receipt belongs to wrong case/child scope')
        packet = json.loads(fields[2])
        if (not isinstance(packet, dict) or set(packet) != {'schema_version', 'reader', 'role', 'complete', 'items'}
                or type(packet['schema_version']) is not int or packet['schema_version'] != 1 or packet['complete'] is not True
                or not isinstance(packet['reader'], str) or not isinstance(packet['role'], str)
                or packet['reader'] in VERIFICATION_ONLY or readers.get(packet['reader']) != packet['role']):
            raise ValueError('input receipt is not a complete source-whitelisted successful reader/role')
        items = packet['items']
        if not isinstance(items, list) or not items:
            raise ValueError('empty actual input collection is not required-work evidence')
        identities = [validate_item(item) for item in items]
        supported = READER_GRADES[packet['reader']]
        for _, kind, _ in identities:
            needed = {'successful-reader-binding': 'reader-binding', 'same-read-buffer': 'content',
                      'validated-content': 'validated-content'}[kind]
            if needed not in supported:
                raise ValueError('input receipt grade exceeds actual source reader consumption capability')
        if len({identity for identity, _, _ in identities}) != len(identities):
            if packet['reader'] not in ORDERED_RECORD_READERS:
                raise ValueError('input collection contains duplicate/partial item identities')
            # Exact repeated records are re-validated against the declared source bounds
            # once the whole contract is known; padding is never accepted by default.
        # An identifier digest is for duplicate handling ONLY; it is never content provenance.
        key = (fields[1], packet['reader'], packet['role'], tuple(identities))
        if key in seen:
            continue
        seen.add(key)
        receipts.append({'scope': fields[1], **packet})
    return receipts


def _assess_inputs(row: dict, text: str, *, observed_requirements: list[dict] | None = None,
                   supporting_pairs: set[tuple[str, str]] | None = None) -> dict:
    contract = validate_contract(row, required=True)
    readers = {**READERS, **contract.get('reader_roles', {})}
    root = row['fqn']
    receipts = _observed_receipts(row, text, readers)
    units = []
    groups = []
    branches = []
    residual_optionals = []
    covered_external = set()
    # An ordered-record producer's repeated lexical records are accepted only as one whole
    # collection of exactly the source-derived arity that INDEPENDENTLY satisfies an
    # applicable declared requirement (reader/role/scope/condition/grade). A duplicate-bearing
    # receipt that is merely below some larger bound elsewhere, or eligible for no declared
    # slot at all, rejects the whole observation exactly like any unflagged duplicate.
    for receipt in receipts:
        identities = [validate_item(item) for item in receipt['items']]
        if len(set(identities)) == len(identities):
            continue
        arity = ORDERED_RECORD_ARITY.get(receipt['reader'])
        if arity is None or len(identities) != arity:
            raise ValueError('ordered-record multiplicity differs from the source producer arity')
        if not any(_receipt_satisfies(requirement, receipt, row) for requirement in contract['requirements']):
            raise ValueError('a duplicate-bearing receipt meets no applicable declared input contract')
    for requirement in contract['requirements']:
        required_scopes = ([root + '#child=' + child for child in row['expected_child_ids']]
                           if requirement['scope'] == 'each-child' else [root])
        for scope in required_scopes:
            external = None
            if requirement.get('when') == 'external-present':
                if any('when' not in other and other['role'] == requirement['role']
                       and other['scope'] == requirement['scope']
                       and set(other['readers']) & set(requirement['readers'])
                       for other in contract['requirements']):
                    # Same API can read both required and optional inputs. Required minimum
                    # units are assigned FIRST; only additional disjoint observations count.
                    residual_optionals.append((requirement, scope))
                    continue
                # Presence is decided by the actual observation alone: an observed optional
                # external input is a real obligation that must be satisfied exactly, while
                # absence imposes nothing. No environment requery, path heuristic, default
                # receipt, or success claim is fabricated for the absent branch.
                present = {index for index, receipt in enumerate(receipts) if receipt['scope'] == scope
                           and receipt['reader'] in requirement['readers'] and receipt['role'] == requirement['role']}
                branches.append({'slot': requirement['slot'], 'scope': scope,
                                 'source_origin': 'external-present' if present else 'external-absent',
                                 'source_origin_basis': 'actual-reader-tag'})
                if not present:
                    continue
                choices = sorted(present)
                if any(not _receipt_satisfies(requirement, receipts[index], row) for index in present):
                    raise ValueError('an observed optional external input is incomplete or below its declared grade')
                covered_external.update(choices)
                first = len(units)
                units.extend([choices] * requirement['minimum_collections'])
                groups.append((requirement, scope, range(first, len(units))))
                continue
            if requirement.get('when') == 'external-selected':
                relevant = [(index, receipt) for index, receipt in enumerate(receipts) if receipt['scope'] == scope
                            and receipt['reader'] in requirement['readers'] and receipt['role'] == requirement['role']]
                if any(item.get('source_origin') not in ('source-defined-default', 'external-selected')
                       for _, receipt in relevant for item in receipt['items']):
                    raise ValueError('conditional input origin is unresolved; cannot infer source default from path/env')
                external = {index for index, receipt in relevant
                            if any(item['source_origin'] == 'external-selected' for item in receipt['items'])}
                # No default event is fabricated. This default is derived only from the
                # explicit frozen branch contract; the caller must separately prove the
                # completed immutable body. Neither environment nor path absence is consulted.
                branches.append({'slot': requirement['slot'], 'scope': scope,
                                 'source_origin': 'external-selected' if external else 'source-defined-default',
                                 'source_origin_basis': 'actual-reader-tag' if external else 'source-defined-branch-contract'})
                if not external:
                    continue
            choices = []
            for index, receipt in enumerate(receipts):
                if external is not None and index not in external:
                    continue
                if receipt['scope'] != scope or receipt['reader'] not in requirement['readers'] or receipt['role'] != requirement['role']:
                    continue
                if _receipt_satisfies(requirement, receipt, row):
                    choices.append(index)
            if external is not None and set(choices) != external:
                raise ValueError('an actually consumed external override has incomplete or insufficient-grade inputs')
            covered_external.update(choices)
            first = len(units)
            units.extend([choices] * requirement['minimum_collections'])
            groups.append((requirement, scope, range(first, len(units))))
    actual_external = {index for index, receipt in enumerate(receipts)
                       if any(item.get('source_origin') == 'external-selected' for item in receipt['items'])}
    if observed_requirements:
        # Alternative-mode observations remain present and graded, even when another complete
        # bundle supplies the required work (for example SIDE+JIN after an empty sidecar read).
        covered_external.update(index for index in actual_external
                                if any(_receipt_satisfies(requirement, receipts[index], row)
                                       for requirement in observed_requirements))
    if supporting_pairs:
        covered_external.update(index for index in actual_external
                                if (receipts[index]['reader'], receipts[index]['role']) in supporting_pairs)
    # Exact matching prevents one receipt satisfying two required collections and avoids greedy
    # rejection when a more-specific pinned slot needs an otherwise eligible receipt.
    owners = {}
    def assign(unit: int, seen_receipts: set[int]) -> bool:
        for receipt in units[unit]:
            if receipt in seen_receipts:
                continue
            seen_receipts.add(receipt)
            if receipt not in owners or assign(owners[receipt], seen_receipts):
                owners[receipt] = unit
                return True
        return False
    # Reserve required-compatible observations that cannot satisfy any colliding optional
    # before filling the other required units. This prevents a strong optional receipt from
    # being stolen by a weak required unit merely because it appeared first in the log.
    invalid_residual = {index for index, receipt in enumerate(receipts)
                        if any(receipt['scope'] == scope and receipt['reader'] in req['readers']
                               and receipt['role'] == req['role'] for req, scope in residual_optionals)
                        and not any(_receipt_satisfies(req, receipt, row) for req, scope in residual_optionals
                                    if receipt['scope'] == scope)}
    if invalid_residual:
        original_units = units
        units = [[index for index in choices if index in invalid_residual] for choices in original_units]
        for unit in range(len(units)):
            assign(unit, set())
        units = original_units
        if invalid_residual - set(owners):
            raise ValueError('a residual optional observation is incomplete or below its declared grade')
    for unit in range(len(units)):
        if unit not in owners.values() and not assign(unit, set()):
            raise ValueError('actual required input slot/grade/complete collection is missing or partial')
    identities = [{validate_item(item)[0] for item in receipt['items']} for receipt in receipts]
    required_items = set().union(*(identities[index] for index in owners)) if owners else set()
    for requirement, scope in residual_optionals:
        choices = []
        for index, receipt in enumerate(receipts):
            if (receipt['scope'] != scope or receipt['reader'] not in requirement['readers']
                    or receipt['role'] != requirement['role'] or identities[index] <= required_items):
                continue
            if identities[index] & required_items:
                raise ValueError('a residual optional collection overlaps required input items')
            if not _receipt_satisfies(requirement, receipt, row):
                raise ValueError('a residual optional observation is incomplete or below its declared grade')
            choices.append(index)
        branches.append({'slot': requirement['slot'], 'scope': scope,
                         'source_origin': 'external-present' if choices else 'not-observed',
                         'source_origin_basis': 'residual-distinct-observation',
                         'selection_not_attested': True})
        if not choices:
            continue
        covered_external.update(choices)
        first = len(units)
        units.extend([choices] * requirement['minimum_collections'])
        groups.append((requirement, scope, range(first, len(units))))
        for unit in range(first, len(units)):
            if not assign(unit, set()):
                raise ValueError('residual distinct optional collection is missing or partial')
    if (contract['fixture_kind'] == 'mixed' or contract.get('source_defined_default') is True) and actual_external - covered_external:
        raise ValueError('an actual consumed external branch has no complete declared input slot')
    assigned = {unit: receipt for receipt, unit in owners.items()}
    matched = [{'slot': requirement['slot'], 'scope': scope,
                'receipt_indices': [assigned[unit] for unit in members], 'required_grade': requirement['required_grade']}
               for requirement, scope, members in groups]
    required_grades = {item['provenance']['kind'] for index in owners for item in receipts[index]['items']}
    # Extra observed actual inputs cannot be hidden to upgrade a receipt-wide content claim.
    grades = {item['provenance']['kind'] for receipt in receipts for item in receipt['items']
              if item.get('source_origin') != 'source-defined-default'}
    weak = 'successful-reader-binding' in grades
    source_default = (contract['fixture_kind'] == 'source-defined' or
                      contract.get('source_defined_default') is True and not groups)
    disjoint_optionals_only = (contract.get('source_defined_default') is True
                               and any(item.get('when') == 'external-present' for item in contract['requirements'])
                               and not any('when' not in item for item in contract['requirements']))
    if not groups and (contract.get('conditional_only') is True or disjoint_optionals_only):
        return {'acceptance_scope': 'no-input-executed', 'input_grade': 'no-input-executed',
                'input_grades': sorted(grades),
                'input_claim_limits': ['external-absent; no required input executed; not successful case execution'],
                'required_input_content_attested': False, 'checkpoint_byte_integrity_attested': False,
                'verified_named_revision': False, 'terminal_model_evidence': False,
                'actual_inputs': receipts, 'required_input_slots': [], 'input_source_branches': branches}
    grade = ('source-defined' if source_default else
             'successful-reader-binding' if weak else 'same-read-buffer' if 'same-read-buffer' in grades else 'validated-content')
    limits = ['case-execution-only; not standalone terminal model evidence']
    if residual_optionals:
        limits += ['same-pair optional presence means residual distinct observation only; branch/caller selection not attested',
                   'optional I/O failure versus absence is not distinguishable at observation layer; original run pass/fail remains mandatory']
    if weak:
        limits += ['checkpoint-byte-integrity-not-attested', 'verified-named-revision-not-claimed']
    elif grade == 'source-defined':
        limits += ['source-defined fixture binding; no external checkpoint integrity claim']
    else:
        limits += ['existing required content pins preserved; no broader named revision claim than those pins']
    return {'acceptance_scope': 'source-bound-case-execution' if weak or grade == 'source-defined' else 'content-bound-case-execution',
            'input_grade': grade, 'input_grades': sorted(grades), 'input_claim_limits': limits,
            'required_input_content_attested': bool(required_grades) and 'successful-reader-binding' not in required_grades,
            'checkpoint_byte_integrity_attested': (not weak and any(requirement['role'] == 'model-weights' for requirement in contract['requirements'])
                and all(requirement.get('expected_sha256') and requirement['required_grade'] != 'reader-binding'
                        for requirement in contract['requirements'] if requirement['role'] == 'model-weights')),
            'verified_named_revision': False, 'terminal_model_evidence': False,
            'actual_inputs': receipts, 'required_input_slots': matched, 'input_source_branches': branches}
