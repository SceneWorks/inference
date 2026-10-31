"""Pure metadata/mocks: never construct tensors or launch a model/GPU query."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci import model_inference_proofs as proof
from scripts.ci import training_comparison as comparison
from scripts.ci.model_proof_resources import bound_cuda_activity_profile


class ComparisonContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = proof.load_registry()
        cls.cases = {comparison.spec(r)['model']: r for r in cls.rows
                     if r['fqn'].split('::')[-1] in comparison._CASES}

    def row(self, model='sdxl', stage='current', binding='gpu0'):
        base = proof.configured_case(self.cases[model], 'fixed_cuda_comparison_' + binding)
        return comparison.stage_row(base, stage)

    def request(self, model='sdxl'):
        base = proof.configured_case(self.cases[model], 'fixed_cuda_comparison_gpu0')
        plan = comparison.spec(base)
        return base, {'selection': proof.identity(base), 'configuration': 'fixed_cuda_comparison_gpu0',
                      'policy': {'backend': 'windows-cuda', 'cudaDeviceUuid': comparison.DEVICES['gpu0'][0]},
                      'training_comparison': {k: plan[k] for k in ('protocol', 'stages', 'max_physical_updates')}}

    def fixture(self, row):
        plan = comparison.spec(row)
        stage, n, warm = row['training_comparison_stage'], plan['updates_per_stage'], plan['warmups_per_stage']
        provider = comparison.case_manifest(row)
        owners = provider.get('owners') or provider.get('owner')
        owners = owners if isinstance(owners, list) else [owners]
        private = plan['model'].startswith('iris-')
        initialization = [{'seed': 7, 'config': 'actual source canonical fixture mock'}]
        if private:
            initialization.append({'kind': 'actual-named-initial-tensors', 'owner': 'private-optimizer', 'sha256': 'c'*64, 'tensor_count': 1})
        value = {'optimizer_updates': n, 'preparation_including_load_cache_seconds': 1.,
                 'comparison': {'schema_version': 1, 'model': plan['model'], 'stage': stage,
                                'cadence': plan['cadence'], 'state_reset': 'fresh-native-child-process',
                                'cuda_visible_devices': row['required_cuda_device_uuid'],
                                'physical_index': row['required_cuda_physical_index'],
                                'physical_starts': n, 'physical_commits': n,
                                'initialization': initialization, 'initial_trainable_fingerprints': ['a'*64],
                                'optimizer_initialization': [] if private else [{'sha256': 'b'*64}],
                                'actual_shared_optimizer_attempts': 0 if private else n}, 'steps': []}
        lines = [f'TRAINING_COMPARISON_STAGE model={plan["model"]} stage={stage} cadence={plan["cadence"]} updates={n}']
        for i in range(1, n+1):
            mechanism = {}
            if plan['applicability']['e1_combined_arithmetic_norm']:
                path = 'legacy' if stage in ('legacy', 'minus-e1') else 'current'
                mechanism['comparison:e1-factor:' + path] = 2
            for group, witnesses in zip(('e2', 'e3'), comparison._WITNESSES[plan['model']], strict=True):
                if witnesses and plan['applicability'][group] and stage not in ('legacy', 'minus-' + group):
                    mechanism[witnesses[0]] = 2
            candidate = int(stage == 'plus-streaming' and not (plan['cadence'] == 'iris-restoration' and i % 2 == 0))
            value['steps'].append({'update': i, 'owner': owners[(i-1) % len(owners)], 'warmup': i <= warm,
                                   'seconds': 10., 'loss': 1. + i/100., 'finite_gradients': 3,
                                   'nonzero_gradients': 2, 'adapter_changed': True,
                                   'attention_candidate': candidate, 'attention_reference': int(candidate == 0),
                                   'mechanism_consumption': mechanism})
            lines += [f'COHORT_UPDATE_START attempted_update={i}', f'COHORT_UPDATE_COMMITTED optimizer_updates={i}']
        return value, '\n'.join(lines) + '\n'

    def test_exact_seventeen_plans_and_404_cap(self):
        self.assertEqual(len(self.cases), 17)
        plans = [comparison.spec(r) for r in self.cases.values()]
        self.assertEqual(sum(x['max_physical_updates'] for x in plans), 404)
        self.assertFalse(any('minus-reference-owner' in p['stages'] for p in plans))
        self.assertEqual(comparison.spec(self.cases['iris-depth'])['stages'], ['legacy', 'current', 'plus-streaming'])
        self.assertNotIn('minus-e3', comparison.spec(self.cases['lens'])['stages'])
        self.assertNotIn('plus-streaming', comparison.spec(self.cases['z_image_turbo'])['stages'])

    def test_every_source_stage_and_owner_cadence(self):
        for model, base in self.cases.items():
            for stage in comparison.spec(base)['stages']:
                with self.subTest(model=model, stage=stage):
                    row = self.row(model, stage)
                    metrics, text = self.fixture(row)
                    result = comparison.verify_metrics(row, metrics, text)
                    self.assertEqual(result['physical_updates'], comparison.spec(base)['updates_per_stage'])

    def test_actual_device_bindings_and_old_default(self):
        for binding in ('gpu0', 'gpu1'):
            row = self.row(binding=binding)
            env = {}
            proof.apply_configuration(row, env)
            self.assertEqual(env['SCENEWORKS_GPU_ID'], str(comparison.DEVICES[binding][1]))
            recorded=proof.configuration_bindings(row,env)
            self.assertTrue(all(recorded.get(k)==v for k,v in proof.expected_configuration_environment(row).items()))
            self.assertEqual(recorded['MODEL_PROOF_COMPARISON_STAGE'],{'present':True,'value':'current'})
            self.assertEqual(bound_cuda_activity_profile(row), proof.SHARED_CUDA_HEADROOM_PROFILE)
        old = proof.configured_case(self.cases['sdxl'], 'fixed_cuda')
        env = {'MODEL_PROOF_COMPARISON_STAGE': 'legacy', 'MODEL_PROOF_COMPARISON_BINDING': 'gpu0'}
        proof.apply_configuration(old, env)
        self.assertNotIn('MODEL_PROOF_COMPARISON_STAGE', env)
        self.assertEqual(old['required_cuda_physical_index'], 1)

    def test_request_is_exact_not_generic_campaign(self):
        row, value = self.request()
        self.assertEqual(comparison.validate_request((row,), value, 'candle-cuda')['max_physical_updates'], 24)
        mutations = []
        for key, val in [('max_physical_updates', 28), ('stages', ['current']), ('protocol', 'other')]:
            changed = copy.deepcopy(value); changed['training_comparison'][key] = val; mutations.append(changed)
        changed = copy.deepcopy(value); changed['policy']['cudaDeviceUuid'] = comparison.DEVICES['gpu1'][0]; mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(ValueError): comparison.validate_request((row,), changed, 'candle-cuda')
        krea = next(r for r in self.rows if r['package'] == 'candle-gen-krea')
        with self.assertRaises(ValueError): comparison.spec(krea)

    def test_wrong_path_hidden_extra_and_truncated_body_rejected(self):
        row = self.row()
        metrics, text = self.fixture(row)
        mutations = [text.replace('stage=current', 'stage=legacy'), text + text,
                     text + 'COHORT_UPDATE_START attempted_update=5\n',
                     text.replace('COHORT_UPDATE_COMMITTED optimizer_updates=4\n', '')]
        for changed in mutations:
            with self.assertRaises(ValueError): comparison.verify_metrics(row, metrics, changed)
        for mutate in (
            lambda m: m['steps'].pop(),
            lambda m: m['steps'][0].update(warmup=False),
            lambda m: m['steps'][0].update(attention_candidate=1),
            lambda m: m['steps'][1].update(owner='other'),
            lambda m: m['steps'][0].update(loss=float('nan')),
            lambda m: m['steps'][0].update(finite_gradients=0),
            lambda m: m['comparison'].update(physical_starts=5),
            lambda m: m['comparison'].update(actual_shared_optimizer_attempts=5),
            lambda m: m['comparison'].update(initial_trainable_fingerprints=[]),
            lambda m: m['comparison'].update(optimizer_initialization=[]),
            lambda m: m['comparison'].update(cuda_visible_devices=comparison.DEVICES['gpu1'][0]),
        ):
            changed = copy.deepcopy(metrics); mutate(changed)
            with self.assertRaises(ValueError): comparison.verify_metrics(row, changed, text)

    def test_actual_off_and_candidate_witnesses_required(self):
        for stage in ('legacy', 'minus-e1', 'minus-e2', 'minus-e3', 'plus-streaming'):
            row = self.row(stage=stage); metrics, text = self.fixture(row)
            if stage == 'plus-streaming':
                metrics['steps'][0]['attention_candidate'] = 0
            elif stage in ('legacy', 'minus-e1'):
                metrics['steps'][0]['mechanism_consumption']['comparison:e1-factor:current'] = 1
            else:
                group = 0 if stage == 'minus-e2' else 1
                metrics['steps'][0]['mechanism_consumption'][comparison._WITNESSES['sdxl'][group][0]] = 1
            with self.assertRaises(ValueError): comparison.verify_metrics(row, metrics, text)

    def test_dual_owner_missing_cycle_or_swapped_owner_rejected(self):
        for model in ('wan2_2_t2v_14b', 'iris-restoration'):
            row = self.row(model); metrics, text = self.fixture(row)
            metrics['steps'][1]['owner'] = metrics['steps'][0]['owner']
            with self.assertRaises(ValueError): comparison.verify_metrics(row, metrics, text)

    def test_aggregate_state_order_and_small_gain(self):
        plan = comparison.spec(self.cases['sdxl'])
        stages = []
        for stage in plan['execution_order']:
            row = self.row(stage=stage); metrics, text = self.fixture(row)
            if stage == 'legacy':
                for step in metrics['steps']: step['seconds'] = 10.5
            stages.append(comparison.verify_metrics(row, metrics, text))
        result = comparison.aggregate(plan, stages)
        self.assertEqual(result['comparisons'][0]['measured_outcome'], 'win')
        self.assertTrue(result['comparisons'][0]['candidate_go'])
        self.assertLess(result['comparisons'][0]['fractional_time_reduction'], .25)
        self.assertFalse(result['production_promotion'])
        for key, value in [('initial_trainable_fingerprints', ['d'*64]), ('initialization', [{'seed':8}]),
                           ('optimizer_initialization', [{'sha256':'d'*64}])]:
            changed=copy.deepcopy(stages); changed[0][key]=value
            with self.assertRaises(ValueError): comparison.aggregate(plan, changed)
        with self.assertRaises(ValueError): comparison.aggregate(plan, stages[::-1])
        with self.assertRaises(ValueError): comparison.aggregate(plan, stages[:-1])

    def test_failure_charge_never_infers_zero_from_missing_logs(self):
        row=self.row()
        self.assertEqual(comparison.update_accounting(row, {'body_created':False}, '')['conservative_charge'],4)
        case={'accepted':False,'cuda_claim_diagnostic':{'requested_uuid':row['required_cuda_device_uuid']},
              'physical_execution':{'mechanism':'metadata-only discovery; no model execution'},
              'error':'claimed CUDA UUID is absent or occupied; no model execution',
              'commands':[{'argv':proof.command(row,True),'supervision':{'returncode':0}}]}
        self.assertTrue(comparison.update_accounting(row,case,'')['zero_source_proven'])
        self.assertFalse(comparison.update_accounting(row,case,'COHORT_UPDATE_START attempted_update=1\n')['zero_source_proven'])

    def mocked_driver(self, *, fail_first=False, inspect=None):
        base,value=self.request()
        value['policy'].update(deadlineSeconds=1000.,pollMillis=100,termGraceMillis=100)
        observed=[]
        def native_mock(rows,backend,child,expected,destination,*,comparison_work):
            row=rows[0]
            observed.append((row['training_comparison_stage'],child['policy']['deadlineSeconds'],str(comparison_work)))
            destination.mkdir(parents=True)
            (destination/'00000').mkdir()
            metrics,text=self.fixture(row)
            result=comparison.verify_metrics(row,metrics,text)
            case={'accepted':not fail_first,'training_comparison':result,
                  'comparison_executable_sha256':'e'*64}
            (destination/'receipt.json').write_text(json.dumps({'accepted':not fail_first,'cases':[case]}))
            (destination/'00000/body.stdout.log').write_text(text if not fail_first else '')
            return 1 if fail_first else 0
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'proof'
            clock=iter(float(x) for x in range(100,1000,2))
            with patch.dict(os.environ,{'RUNNER_TEMP':directory}), \
                 patch.object(proof,'repository_identity',return_value={'revision':'f'*40}), \
                 patch.object(proof,'run',side_effect=native_mock), \
                 patch.object(proof,'verify_artifacts',return_value={}), \
                 patch.object(comparison.time,'monotonic',side_effect=lambda:next(clock)):
                code=comparison.run((base,),'candle-cuda',value,'f'*40,root)
            receipt=json.loads((root/'receipt.json').read_text())
            self.assertEqual(receipt['artifacts'],comparison._inventory(root))
            if inspect is not None:
                inspect(root,value,base,receipt)
            return code,receipt,observed

    def test_driver_one_fresh_namespace_each_stage_same_original_deadline(self):
        code,receipt,observed=self.mocked_driver()
        self.assertEqual(code,0)
        self.assertEqual([x[0] for x in observed],receipt['plan']['execution_order'])
        self.assertEqual(len(set(x[2] for x in observed)),len(observed))
        self.assertTrue(all(a[1]>b[1] for a,b in zip(observed,observed[1:])))
        for record in receipt['stage_records']:
            self.assertEqual(record['remaining_original_deadline_seconds'],1000.-record['elapsed_before_stage_seconds'])
        self.assertEqual(receipt['conservative_updates_charged'],24)

    def test_driver_stops_and_reserves_failed_unknown_stage_without_retry(self):
        code,receipt,observed=self.mocked_driver(fail_first=True)
        self.assertEqual(code,1)
        self.assertEqual(len(observed),1)
        self.assertFalse(receipt['accepted'])
        self.assertEqual(receipt['conservative_updates_charged'],4)
        self.assertNotIn('measurement',receipt)

    def test_shared_source_policy_rejects_math_counter_only_mutations(self):
        root=Path(__file__).resolve().parents[2]/'crates/media/candle-gen/candle-gen/src'
        sources={name:(root/('train/' if name!='reference_training_attention.rs' else '')/name).read_text(encoding='utf-8')
                 for name in ('projection.rs','norm.rs','reference_training_attention.rs','profile.rs','optim.rs')}
        comparison.validate_shared_source_controls(sources)
        mutations=(
            ('projection.rs','return x.broadcast_matmul(w);','return x.matmul(w);'),
            ('projection.rs','return apply_factor(&rank, up)? * scale;','return apply_factor(&(rank * scale)?, up);'),
            ('norm.rs','return normalize_composable(x, eps, reference);','return normalize_composable(&x.detach(), eps, reference);'),
            ('norm.rs','xf.broadcast_div(&rms)','xf.broadcast_mul(&rms.recip()?)'),
            ('reference_training_attention.rs','reference_owner_enabled()','cache_enabled()'),
            ('optim.rs','super::profile::comparison_optimizer_step()?;',''),
        )
        for name,old,new in mutations:
            self.assertIn(old,sources[name])
            changed=dict(sources); changed[name]=changed[name].replace(old,new,1)
            with self.subTest(file=name,old=old),self.assertRaises(ValueError):
                comparison.validate_shared_source_controls(changed)

    def test_aggregate_verifier_authenticates_stage_join_deadline_and_inventory(self):
        def inspect(root,value,base,receipt):
            def verify():
                return comparison.verify_artifacts(root,root.parent,value,'f'*40,(base,),'candle-cuda')
            with patch.object(proof,'verify_artifacts',return_value={}):
                self.assertTrue(verify()['accepted'])
                for mutate in (
                    lambda x:x['stage_records'].pop(),
                    lambda x:x['stage_records'][0].update(remaining_original_deadline_seconds=1000.),
                    lambda x:x['stage_records'][0].update(executable_sha256='d'*64),
                    lambda x:x.update(conservative_updates_charged=0),
                    lambda x:x.update(terminal_model_evidence=True),
                ):
                    changed=copy.deepcopy(receipt); mutate(changed)
                    (root/'receipt.json').write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):verify()
                (root/'receipt.json').write_text(json.dumps(receipt))
                extra=root/'stages'/'unexpected.txt';extra.write_text('unapproved')
                with self.assertRaises(ValueError):verify()
                extra.unlink()
                child=root/receipt['stage_records'][0]['artifact']/'receipt.json'
                child.write_text(child.read_text()+'\n')
                with self.assertRaises(ValueError):verify()
        code,_,_=self.mocked_driver(inspect=inspect)
        self.assertEqual(code,0)


if __name__ == '__main__':
    unittest.main()
