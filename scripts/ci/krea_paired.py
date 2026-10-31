"""Scalar-only protocol for one source-owned original/candidate Krea comparison.

No old artifact or baseline is imported/relabelled. Failed or partial bodies never
pass; durable update-start records charge attempted optimizer work conservatively.
"""
from __future__ import annotations

import math
import re
import struct

from scripts.ci import krea_recovery as recovery

CONFIGURATION = 'fixed_cuda_paired_comparison'
SELECTOR = 'paired-12'
GUARD = 'original0-six;candidate2-six;skip-shared1;prior-charge20;hard-cap32;one-invocation'
ADAPTER_ROLE = 'actual-paired-candidate-even-on-no-go'
COMPARISON = 'same-process-original-reference-vs-full-candidate'
CLAIM_LIMIT = 'One serial shared-GPU invocation, not an isolated benchmark. Original Reference0 then fully prepared candidate2; segment1 skipped. Identical deep initial factors, fresh zero-state optimizer and stateless seeded logical inputs; shared frozen weights/cache. Two counted warmups each. Preparation and diagnostics excluded from synchronized steady timing. Smoke, adoption, >=25percent target and convergence are separate; no weight byte attestation.'


def validate_binding(row: dict) -> bool:
    if not recovery.applies(row):
        return False
    recipe = row.get('selected_configuration', {})
    if recipe.get('name') != CONFIGURATION:
        return False
    if (row.get('configuration_kind') != 'source-controls'
            or row.get('configuration_controls') != [recovery.CONTROL]
            or recipe.get('environment') != {recovery.CONTROL: SELECTOR}
            or recipe.get('source_guard') != GUARD
            or recipe.get('required_cuda_physical_index') != 1
            or recipe.get('cuda_activity_profile') != 'non-isolated-correctness-headroom'
            or row.get('required_cuda_device_uuid') != 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe'):
        raise ValueError('Krea paired binding requires the exact source-owned twelve-update/GPU1 recipe')
    return True


def _f32(value: float) -> float:
    try:
        return struct.unpack('!f', struct.pack('!f', value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError('paired loss outside finite f32 domain') from error


def attempted_updates(text: str) -> int:
    """Accounting hint only: each durable start counts, even when uncommitted."""
    return len(re.findall(r'^KREA_PROFILE_BOUNDARY phase=update-start(?: |$)', text, re.M))


def update_accounting(text: str | None, *, body_launched: bool, accepted: bool) -> dict:
    starts = attempted_updates(text or '')
    commits = len(re.findall(r'^KREA_PROFILE_BOUNDARY phase=update-committed(?: |$)',text or '',re.M))
    return {'observed_starts':starts,'observed_commits':commits,
            'observed_uncommitted_starts':max(0,starts-commits),
            'prior_conservative_charge':20,'authorized_new_update_cap':12,'hard_total_cap':32,
            'conservative_new_charge':12 if body_launched else 0,
            'complete_accepted_accounting':accepted,
            'claim_limit':'Every warmup/partial/failed start counts. Incomplete/missing logs are lower bounds; reserve all12 for a launched failed body until separately audited. No automatic retry.'}


def verify_execution(row: dict, text: str) -> dict:
    if not recovery.applies(row):
        return {}
    if not validate_binding(row):
        if 'KREA_PROFILE_PAIRED_' in text:
            raise ValueError('paired evidence cannot masquerade as six/eighteen updates')
        return {}
    if 'KREA_PROFILE_RECOVERY_' in text or re.search(r'^KREA_PROFILE_RESULT ', text, re.M):
        raise ValueError('paired comparison cannot reuse old recovery or eighteen-update evidence')
    expected = [(str(i), str(0 if i <= 6 else 2), str((i-1)%6+1)) for i in range(1,13)]
    boundaries = re.findall(r'^KREA_PROFILE_BOUNDARY phase=(update-[^ ]+) physical_update=(\d+) segment=(\d+) logical_step=(\d+)$', text, re.M)
    if boundaries != [(phase,*step) for step in expected for phase in ('update-start','update-committed')] or attempted_updates(text)!=12 or text.count('KREA_PROFILE_BOUNDARY phase=update-')!=24:
        raise ValueError('paired comparison requires ordered twelve starts/commits, six per path, no extras')
    steps = re.findall(r'^KREA_PROFILE_STEP physical_update=(\d+)/(\d+) segment=(\d+) logical_step=(\d+) warmup=(true|false) seconds=([^ ]+) loss=([^ ]+) attention_candidate=(\d+) attention_reference=(\d+)$', text, re.M)
    gradients = re.findall(r'^KREA_PROFILE_GRADIENTS physical_update=(\d+)/(\d+) finite=(\d+) total=(\d+)$', text, re.M)
    if (len(steps)!=12 or text.count('KREA_PROFILE_STEP ')!=12
            or gradients != [(str(i),'12','224','224') for i in range(1,13)]
            or text.count('KREA_PROFILE_GRADIENTS ')!=12):
        raise ValueError('paired comparison requires twelve actual step and all224 finite-gradient records')
    # Verify temporal per-step order too: aggregate per-phase counts alone could accept replay/reordering.
    events = re.findall(r'^(KREA_PROFILE_BOUNDARY phase=update-(?:start|committed)|KREA_PROFILE_PAIRED_PATH|KREA_PROFILE_STEP|KREA_PROFILE_GRADIENTS) .*$', text, re.M)
    if events != ['KREA_PROFILE_BOUNDARY phase=update-start','KREA_PROFILE_PAIRED_PATH','KREA_PROFILE_BOUNDARY phase=update-committed','KREA_PROFILE_STEP','KREA_PROFILE_GRADIENTS']*12:
        raise ValueError('paired update/step/gradient event order mismatch')
    states = re.findall(r'^KREA_PROFILE_PAIRED_STATE segment=([02]) adapter_initial_equal=true parameters=224 optimizer=fresh-zero-moments update_idx=0 rng=stateless-seed7-logical1to6$',text,re.M)
    adapters = re.findall(r'^KREA_PROFILE_PAIRED_ADAPTER segment=([02]) changed=true$',text,re.M)
    if (states!=['0','2'] or adapters!=['0','2'] or text.count('KREA_PROFILE_PAIRED_STATE ')!=2 or text.count('KREA_PROFILE_PAIRED_ADAPTER ')!=2):
        raise ValueError('paired initial-state restoration or actual changed adapters absent')
    for segment, first, last in ((0,1,6),(2,7,12)):
        state = text.index(f'KREA_PROFILE_PAIRED_STATE segment={segment} ')
        start = text.index(f'KREA_PROFILE_BOUNDARY phase=update-start physical_update={first} ')
        finish = text.index(f'KREA_PROFILE_GRADIENTS physical_update={last}/12 ')
        adapter = text.index(f'KREA_PROFILE_PAIRED_ADAPTER segment={segment} ')
        if not state < start < finish < adapter:
            raise ValueError('paired state/adapter witnesses outside their actual work scopes')
    if text.index('KREA_PROFILE_PAIRED_ADAPTER segment=0 ') > text.index('KREA_PROFILE_PAIRED_STATE segment=2 '):
        raise ValueError('candidate state precedes original segment completion')
    attribution = re.findall(r'^KREA_PROFILE_ATTRIBUTION segment=(\d+) synchronized_inclusive_spans=.+$',text,re.M)
    optimizer = re.findall(r'^KREA_PROFILE_OPTIMIZER segment=(\d+) synchronized_seconds=([^ ]+)$',text,re.M)
    preparation = re.findall(r'^KREA_PROFILE_PREPARATION seconds=([^ ]+)$',text,re.M)
    if (attribution!=['0','2'] or text.count('KREA_PROFILE_ATTRIBUTION ')!=2
            or [s for s,_ in optimizer]!=['0','2'] or text.count('KREA_PROFILE_OPTIMIZER ')!=2
            or len(preparation)!=1 or text.count('KREA_PROFILE_PREPARATION ')!=1
            or any(not math.isfinite(float(t)) or float(t)<0 for t in [*(v for _,v in optimizer),*preparation])):
        raise ValueError('paired synchronized warmup attribution/optimizer and separate preparation missing')
    prep=text.index('KREA_PROFILE_PREPARATION ')
    if not text.index('KREA_PROFILE_PAIRED_ADAPTER segment=0 ') < prep < text.index('KREA_PROFILE_PAIRED_STATE segment=2 '):
        raise ValueError('candidate preparation must follow original work and precede candidate state/timing')
    packs = re.findall(r'^krea training: (\d+) frozen Q/K/V/gate packs, (\d+) resident base bytes shared by source views$',text,re.M)
    captions = re.findall(r'^krea training: prepared (\d+) unique captions in .+$',text,re.M)
    if len(packs)!=1 or packs[0][0]!='28' or int(packs[0][1])<=0 or captions!=['2']:
        raise ValueError('paired full candidate requires actual28 packed projections and two prepared captions')
    for marker in ('krea training: 28 frozen Q/K/V/gate packs,','krea training: prepared 2 unique captions in '):
        if not text.index('KREA_PROFILE_PAIRED_ADAPTER segment=0 ') < text.index(marker) < prep:
            raise ValueError('paired preparation inventory outside candidate preparation phase')
    spans = re.findall(r'^KREA_PROFILE_ATTRIBUTION segment=[02] synchronized_inclusive_spans=(.+)$',text,re.M)
    for span in spans:
        counts = re.findall(r'"([^"\n]+)": \((\d+), ([0-9.eE+-]+)\)',span)
        if (not {'conditioning_pre_main','qkvg_projection_forward','norm_forward','attention_forward','whole_backward'} <= {name for name,_,_ in counts}
                or any(int(count)<=0 or not math.isfinite(float(seconds)) or float(seconds)<=0 for _,count,seconds in counts)):
            raise ValueError('paired attribution must preserve actual synchronized operation counts/times')
    report = recovery._record(text,'KREA_PROFILE_PAIRED_RESULT ')
    identity = recovery._record(text,'KREA_PROFILE_PAIRED_EXECUTION ')
    if (set(identity)!={'source','run','attempt'} or any(type(value) is not str for value in identity.values()) or not re.fullmatch(r'[0-9a-f]{40}',str(identity.get('source','')))
            or any(not re.fullmatch(r'[1-9][0-9]*',str(identity.get(key,''))) for key in ('run','attempt'))):
        raise ValueError('paired comparison needs actual source/run/attempt identity')
    fixed = {'schema_version':1,'new_optimizer_updates':12,'segment_order':[0,2],
        'warmup_updates_per_segment':2,'steady_updates_per_segment':4,
        'prior_conservative_update_charge':20,'hard_update_cap':32,'remaining_updates_after_success':0,
        'saved_adapter_role':ADAPTER_ROLE,'comparison':COMPARISON,'claim_limit':CLAIM_LIMIT}
    if any(type(report.get(k)) is not type(v) or report.get(k)!=v for k,v in fixed.items()):
        raise ValueError('paired source budget/order/comparator altered')
    paths = re.findall(r'^KREA_PROFILE_PAIRED_PATH physical_update=(\d+) segment=(\d+) reference_math=(true|false) attention_mode=(Reference|StreamingCandidate)$',text,re.M)
    if (paths != [(str(i),str(0 if i<=6 else 2),'true' if i<=6 else 'false','Reference' if i<=6 else 'StreamingCandidate') for i in range(1,13)]
            or text.count('KREA_PROFILE_PAIRED_PATH ')!=12):
        raise ValueError('paired actual original/reference math or full candidate path witness mismatch')
    observations={0:[],2:[]}
    for i, step in enumerate(steps,1):
        physical,limit,segment,logical,warmup,seconds,loss,candidate,reference=step
        wanted=expected[i-1]
        if ((physical,segment,logical)!=wanted or limit!='12' or warmup!=str(int(logical)<=2).lower()
                or (candidate,reference)!=(('0','28') if segment=='0' else ('28','0'))
                or not math.isfinite(float(seconds)) or float(seconds)<=0 or not math.isfinite(float(loss))):
            raise ValueError('paired actual path/dispatch/step/settings mismatch')
        observations[int(segment)].append((float(seconds),float(loss)))
    means={}
    for segment,prefix in ((0,'reference'),(2,'candidate')):
        times,losses=report.get(prefix+'_steady_seconds'),report.get(prefix+'_losses')
        if (not isinstance(times,list) or len(times)!=4 or not isinstance(losses,list) or len(losses)!=6
                or any(type(x) not in (int,float) or not math.isfinite(x) for x in [*times,*losses])
                or any(x<=0 for x in times) or any(x != _f32(x) for x in losses)):
            raise ValueError('paired report requires actual six losses/four positive synchronized timings per path')
        for j,(seconds,loss) in enumerate(observations[segment]):
            if _f32(loss)!=losses[j] or (j>=2 and not recovery._close(times[j-2],seconds,.000000500001)):
                raise ValueError('paired scalar report differs from actual rounded step observations')
        means[segment]=sum(times)/4
        if not recovery._close(report.get(prefix+'_mean_seconds'),means[segment]):
            raise ValueError('paired mean differs from actual four timed observations')
    # Match Rust f32 subtraction/max/multiply for the inherited all-six loss-parity decision.
    parity=all(abs(_f32(a-b))<=_f32(_f32(.01)*max(abs(a),1.)) for a,b in zip(report['reference_losses'],report['candidate_losses']))
    go=parity and means[2]<means[0]
    decisions={'candidate_dispatches':168,'loss_parity':parity,'candidate_go':go,'target_met':go and means[2]<=.75*means[0]}
    if (any(type(report.get(k)) is not type(v) or report.get(k)!=v for k,v in decisions.items())
            or not recovery._close(report.get('candidate_reduction_vs_original'),1-means[2]/means[0])
            or not recovery._close(report.get('candidate_latency_ratio'),means[2]/means[0])
            or not recovery._close(report.get('candidate_seconds_saved_per_update'),means[0]-means[2])):
        raise ValueError('paired decisions must be recomputed from original0 versus candidate2')
    return {'krea_paired':report,'krea_paired_execution':identity,
            'krea_paired_update_accounting':update_accounting(text,body_launched=True,accepted=True)}


def verify_metrics(row: dict, metrics: dict, execution: dict, expected_source: str) -> None:
    if not validate_binding(row):
        return
    report,identity=execution['krea_paired'],execution['krea_paired_execution']
    expected={'rank':16,'alpha':16.,'learning_rate':struct.unpack('!f',struct.pack('!f',.001))[0],
        'optimizer':'adamw','batch_size':1,'resolution':1024,'train_dtype':'bf16','gradient_accumulation':1,
        'gradient_checkpointing':False,'seed':7,'sample_every':500,'save_every':0}
    if (type(metrics.get('completed_optimizer_updates')) is not int or metrics.get('completed_optimizer_updates')!=12 or metrics.get('paired_comparison')!=report
            or metrics.get('paired_execution')!=identity or identity['source']!=expected_source
            or metrics.get('claim_limit')!=CLAIM_LIMIT or metrics.get('saved_adapter_role')!=ADAPTER_ROLE or metrics.get('graph_retaining_backward') is not False
            or metrics.get('losses_by_physical_update')!=report['reference_losses']+report['candidate_losses']
            or metrics.get('selected_adapter_final_loss')!=report['candidate_losses'][-1]
            or metrics.get('sample_configuration')!={'width':1024,'height':1024,'count':1,'seed':7,'denoise_steps':28}
            or any(type(metrics.get('configuration',{}).get(k)) is not type(v) or metrics['configuration'].get(k)!=v for k,v in expected.items())):
        raise ValueError('paired metrics/source/settings/actual candidate lifecycle mismatch')
