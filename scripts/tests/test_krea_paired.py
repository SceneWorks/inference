"""Pure metadata/mocked Krea paired protocol and mutation checks; no model work."""
import copy
import json
from pathlib import Path
import struct
import unittest

from scripts.ci import krea_paired as paired
from scripts.ci import krea_recovery as recovery
from scripts.ci import model_inference_proofs as proof

ROOT=Path(__file__).resolve().parents[2]


def fixture(candidate_seconds=6., loss_change=0.):
    source=next(row for row in proof.load_registry() if recovery.applies(row))
    row=proof.configured_case(source,paired.CONFIGURATION)
    f32=lambda value: struct.unpack('!f',struct.pack('!f',value))[0]
    original=[f32(.8-i*.02) for i in range(6)]
    candidate=[f32(loss+loss_change) for loss in original]
    parity=all(abs(f32(a-b))<=f32(f32(.01)*max(abs(a),1.)) for a,b in zip(original,candidate))
    go=parity and candidate_seconds<10.
    report={'schema_version':1,'new_optimizer_updates':12,'segment_order':[0,2],
        'warmup_updates_per_segment':2,'steady_updates_per_segment':4,
        'prior_conservative_update_charge':20,'hard_update_cap':32,'remaining_updates_after_success':0,
        'reference_steady_seconds':[10.]*4,'candidate_steady_seconds':[candidate_seconds]*4,
        'reference_losses':original,'candidate_losses':candidate,
        'reference_mean_seconds':10.,'candidate_mean_seconds':candidate_seconds,
        'candidate_dispatches':168,'loss_parity':parity,'candidate_go':go,
        'candidate_reduction_vs_original':1-candidate_seconds/10.,
        'candidate_latency_ratio':candidate_seconds/10.,'candidate_seconds_saved_per_update':10.-candidate_seconds,
        'target_met':go and candidate_seconds<=7.5,'saved_adapter_role':paired.ADAPTER_ROLE,'comparison':paired.COMPARISON,'claim_limit':paired.CLAIM_LIMIT}
    identity={'source':'1'*40,'run':'41200000000','attempt':'1'}
    lines=['test result: ok. 1 passed; 0 failed; 0 ignored;',
        'MODEL_INFERENCE_BODY_COMPLETED '+recovery.FQN,
        'MODEL_INFERENCE_DEVICE '+recovery.FQN+' candle-cuda']
    for segment,losses,seconds in ((0,original,10.),(2,candidate,candidate_seconds)):
        if segment==2:
            lines.extend(['krea training: 28 frozen Q/K/V/gate packs, 1024 resident base bytes shared by source views',
                'krea training: prepared 2 unique captions in 1ms','KREA_PROFILE_PREPARATION seconds=0.1'])
        lines.append(f'KREA_PROFILE_PAIRED_STATE segment={segment} adapter_initial_equal=true parameters=224 optimizer=fresh-zero-moments update_idx=0 rng=stateless-seed7-logical1to6')
        for logical in range(1,7):
            i=logical+(6 if segment==2 else 0)
            for phase in ('update-start','update-committed'):
                lines.append(f'KREA_PROFILE_BOUNDARY phase={phase} physical_update={i} segment={segment} logical_step={logical}')
            lines.insert(len(lines)-1,f'KREA_PROFILE_PAIRED_PATH physical_update={i} segment={segment} reference_math={str(segment==0).lower()} attention_mode={"Reference" if segment==0 else "StreamingCandidate"}')
            if logical==1:
                lines.append(f'KREA_PROFILE_OPTIMIZER segment={segment} synchronized_seconds=0.01')
            lines.append(f'KREA_PROFILE_STEP physical_update={i}/12 segment={segment} logical_step={logical} warmup={str(logical<=2).lower()} seconds={seconds:.6f} loss={losses[logical-1]} attention_candidate={28 if segment==2 else 0} attention_reference={28 if segment==0 else 0}')
            if logical==1:
                spans=', '.join('"'+name+'": (28, 0.01)' for name in ('conditioning_pre_main','qkvg_projection_forward','norm_forward','attention_forward','whole_backward'))
                lines.append(f'KREA_PROFILE_ATTRIBUTION segment={segment} synchronized_inclusive_spans={{{spans}}}; mock scalar fixture only')
            lines.append(f'KREA_PROFILE_GRADIENTS physical_update={i}/12 finite=224 total=224')
        lines.append(f'KREA_PROFILE_PAIRED_ADAPTER segment={segment} changed=true')
    lines.extend(['KREA_PROFILE_PAIRED_RESULT '+json.dumps(report),'KREA_PROFILE_PAIRED_EXECUTION '+json.dumps(identity)])
    metrics={'completed_optimizer_updates':12,'paired_comparison':report,'paired_execution':identity,
        'saved_adapter_role':paired.ADAPTER_ROLE,'claim_limit':paired.CLAIM_LIMIT,'graph_retaining_backward':False,
        'losses_by_physical_update':original+candidate,'selected_adapter_final_loss':candidate[-1],
        'sample_configuration':{'width':1024,'height':1024,'count':1,'seed':7,'denoise_steps':28},
        'configuration':{'rank':16,'alpha':16.,'learning_rate':f32(.001),'optimizer':'adamw','batch_size':1,'resolution':1024,
        'train_dtype':'bf16','gradient_accumulation':1,'gradient_checkpointing':False,'seed':7,'sample_every':500,'save_every':0}}
    return row,lines,report,identity,metrics


def log(lines):
    return '\n'.join(lines)+'\n'


class KreaPairedProtocol(unittest.TestCase):
    def test_named_binding_distinguishes_twelve_six_eighteen_and_clears_inherited_selector(self):
        row,*_=fixture()
        env={recovery.CONTROL:paired.SELECTOR}
        proof.apply_configuration(proof.configured_case(row),env)
        self.assertNotIn(recovery.CONTROL,env)
        proof.apply_configuration(row,env)
        self.assertEqual(env[recovery.CONTROL],paired.SELECTOR)
        self.assertTrue(paired.validate_binding(row))
        self.assertFalse(recovery.validate_binding(row))
        for change in (lambda r:r['configurations'][paired.CONFIGURATION]['environment'].update({recovery.CONTROL:'1'}),
                       lambda r:r['configurations'][paired.CONFIGURATION].update(source_guard='18-updates'),
                       lambda r:r.update(required_cuda_device_uuid='GPU-'+'0'*36)):
            changed=copy.deepcopy(row);change(changed)
            with self.assertRaises(ValueError):proof.configured_case(changed,paired.CONFIGURATION)

    def test_complete_mocked_protocol_and_source_bound_metrics(self):
        row,lines,report,identity,metrics=fixture()
        result=proof.verify_execution(row,log(lines))
        self.assertEqual(result['krea_paired'],report)
        self.assertEqual(result['krea_paired_execution'],identity)
        self.assertTrue(report['candidate_go']);self.assertTrue(report['target_met'])
        paired.verify_metrics(row,metrics,result,identity['source'])
        self.assertEqual(result['krea_paired_update_accounting']['observed_starts'],12)

    def test_slower_miss_and_parity_failure_still_preserve_actual_candidate(self):
        for seconds,delta,go,target in ((28.,0.,False,False),(8.,0.,True,False),(6.,.2,False,False),(7.5,0.,True,True)):
            row,lines,report,identity,metrics=fixture(seconds,delta)
            result=proof.verify_execution(row,log(lines))
            paired.verify_metrics(row,metrics,result,identity['source'])
            self.assertEqual((report['candidate_go'],report['target_met']),(go,target))
            self.assertEqual(metrics['saved_adapter_role'],paired.ADAPTER_ROLE)

    def test_mutations_reject_count_order_segments_warmup_gradient_dispatch_state_truncated_body(self):
        row,lines,*_=fixture()
        mutations=[
            lambda xs:xs+['KREA_PROFILE_BOUNDARY phase=update-start physical_update=13 segment=2 logical_step=7'],
            lambda xs:[x for x in xs if 'phase=update-start physical_update=12 ' not in x],
            lambda xs:[x for x in xs if not x.startswith('KREA_PROFILE_GRADIENTS physical_update=12/')],
            lambda xs:[x.replace('segment=2','segment=1') for x in xs],
            lambda xs:[x.replace('/12 ','/18 ') for x in xs],
            lambda xs:[x.replace('warmup=true','warmup=false') for x in xs],
            lambda xs:[x.replace('finite=224 total=224','finite=223 total=224') for x in xs],
            lambda xs:[x.replace('attention_candidate=28','attention_candidate=0') for x in xs],
            lambda xs:[x.replace('adapter_initial_equal=true','adapter_initial_equal=false') for x in xs],
            lambda xs:[x.replace('reference_math=true','reference_math=false') for x in xs],
            lambda xs:[x.replace('attention_mode=StreamingCandidate','attention_mode=Reference') for x in xs],
            lambda xs:[x.replace('28 frozen Q/K/V/gate','0 frozen Q/K/V/gate') for x in xs],
            lambda xs:[x.replace('prepared 2 unique captions','prepared 0 unique captions') for x in xs],
            lambda xs:[x.replace('"whole_backward": (28, 0.01)','"whole_backward": (0, 0.01)') for x in xs],
            lambda xs:[x for x in xs if not x.startswith('KREA_PROFILE_PAIRED_ADAPTER segment=0')],
            lambda xs:[x for x in xs if not x.startswith('KREA_PROFILE_ATTRIBUTION segment=2')],
            lambda xs:[x for x in xs if not x.startswith('KREA_PROFILE_PREPARATION')],
            lambda xs:xs+['KREA_PROFILE_RESULT updates=18'],
            lambda xs:xs+['KREA_PROFILE_RECOVERY_ADAPTER changed=true'],
            lambda xs:[x for x in xs if not x.startswith('MODEL_INFERENCE_BODY_COMPLETED')],
            lambda xs:xs[:-1],
            lambda xs:[x.replace('1 passed; 0 failed','0 passed; 1 failed') for x in xs],
        ]
        def reversed_boundaries(xs):
            indexes=[i for i,x in enumerate(xs) if 'phase=update-' in x]
            xs[indexes[0]],xs[indexes[1]]=xs[indexes[1]],xs[indexes[0]]
            return xs
        mutations.append(reversed_boundaries)
        for mutate in mutations:
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                proof.verify_execution(row,log(mutate(list(lines))))
        for count in (0,12,36,len(lines)-3):
            with self.subTest(truncation=count),self.assertRaises(ValueError):
                proof.verify_execution(row,log(lines[:count]))

    def test_mutations_reject_scalar_decision_and_budget_fabrication(self):
        row,lines,report,*_=fixture()
        for key,value in {'hard_update_cap':33,'prior_conservative_update_charge':19,'segment_order':[1,2],
            'new_optimizer_updates':18,'candidate_dispatches':0,'reference_mean_seconds':9.,'claim_limit':'isolated benchmark',
            'candidate_go':False,'target_met':False,'loss_parity':False,'candidate_reduction_vs_original':.5,'candidate_latency_ratio':.5,'candidate_seconds_saved_per_update':.1,
            'reference_losses':[.5]*6,'candidate_steady_seconds':[1.]*4,'candidate_losses':[1e300]*6}.items():
            changed={**report,key:value}
            mutated=[x if not x.startswith('KREA_PROFILE_PAIRED_RESULT ') else 'KREA_PROFILE_PAIRED_RESULT '+json.dumps(changed) for x in lines]
            with self.subTest(key=key),self.assertRaises(ValueError):proof.verify_execution(row,log(mutated))

    def test_mutations_reject_settings_and_lifecycle_changes_and_f64_lr_literal(self):
        row,lines,report,identity,metrics=fixture()
        execution=proof.verify_execution(row,log(lines))
        for key,value in {'rank':8,'alpha':8.,'learning_rate':.001,'optimizer':'adam','resolution':512,
            'batch_size':2,'train_dtype':'f32','gradient_accumulation':2,'gradient_checkpointing':True,'seed':8,
            'sample_every':0,'save_every':500}.items():
            changed=copy.deepcopy(metrics);changed['configuration'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):paired.verify_metrics(row,changed,execution,identity['source'])
        for key,value in {'completed_optimizer_updates':18,'saved_adapter_role':'reference',
            'selected_adapter_final_loss':.123,'sample_configuration':{'count':2},'losses_by_physical_update':report['candidate_losses']}.items():
            changed={**metrics,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):paired.verify_metrics(row,changed,execution,identity['source'])
        with self.assertRaises(ValueError):paired.verify_metrics(row,metrics,execution,'2'*40)

    def test_failure_accounting_charges_partial_warmup_and_unknown_launched_body(self):
        _,lines,*_=fixture()
        partial=log(lines[:lines.index(next(x for x in lines if 'phase=update-committed physical_update=2 ' in x))])
        counted=paired.update_accounting(partial,body_launched=True,accepted=False)
        self.assertEqual((counted['observed_starts'],counted['observed_commits'],counted['observed_uncommitted_starts']),(2,1,1))
        self.assertEqual(counted['conservative_new_charge'],12)
        self.assertEqual(paired.update_accounting(None,body_launched=True,accepted=False)['conservative_new_charge'],12)
        self.assertEqual(paired.update_accounting(None,body_launched=False,accepted=False)['conservative_new_charge'],0)

    def test_source_admission_and_restoration_precede_work_and_no_tail_update_is_hidden(self):
        source=(ROOT/'crates/media/candle-gen/candle-gen/src/train/flow_match.rs').read_text(encoding='utf-8')
        self.assertLess(source.index('cfg != &expected || req.items.len() != 2'),source.index('model.cache(req, device, on_progress)?'))
        self.assertIn('steps: 12',source);self.assertIn('learning_rate: 0.001_f32',source)
        self.assertIn('if recovery_profile || (paired_profile && step > 6)',source)
        self.assertIn('if segment == 1 || recovery_profile || paired_profile',source)
        state=source.index('paired initial optimizer/accumulation state mismatch')
        self.assertLess(source.index('var.set(value)?'),state)
        self.assertLess(source.index('update_idx = 0;',source.index('var.set(value)?')),state)
        self.assertLess(state,source.index('phase=update-start physical_update={step}'))
        self.assertIn('steps_run != 12 || pending != 0 || accumulated.is_some()',source)
        self.assertIn('step_sample(aux_driver.as_mut(), logical_step, &schedule)',source)
        self.assertIn('segment_cfg.steps = 6;',source)
        self.assertIn('if bounded_profile { 6 } else { cfg.steps }',source)
        trainer=(ROOT/'crates/media/candle-gen/candle-gen-krea/src/training.rs').read_text(encoding='utf-8')
        self.assertIn('flow_match::timestep_seed(cfg.seed, step)',trainer)
        self.assertIn('flow_match::noise_seed(cfg.seed, step)',trainer)


if __name__=='__main__':unittest.main()
