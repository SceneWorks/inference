#!/usr/bin/env python3
"""Sampler arithmetic/negative controls and predeclared source-only perturbation analysis.
Does not modify source math, source fixture or its expected values. No native data is read.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, hashlib, json, copy, inspect
from pathlib import Path
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PARENT=ROOT/'tiny_edit_v2/fixture.json'
parent_bytes=PARENT.read_bytes()
assert hashlib.sha256(parent_bytes).hexdigest()=='398b6b2260c15b670655700ed552d4ee935d493af5c9f2a72c85570cd7d121f1'
original=json.loads(parent_bytes)
# Execute only exact v2 construction definitions, before its capture/output section.
program=(ROOT/'tiny_edit_v2/generate.py').read_text()
assert hashlib.sha256(program.encode()).hexdigest()=='d464fcac7bf3b10989ec43f71f77a6f38340655ad59f618adc4a2745c0ed61f7'
ns0={'__file__':str(ROOT/'tiny_edit_v2/generate.py'),'__name__':'_v2_construction'}
exec(compile(program.split("\nwith torch.no_grad():\n    fixture['time_probes']")[0],str(ROOT/'tiny_edit_v2/generate.py'),'exec'),ns0)
globals().update({k:v for k,v in ns0.items() if k not in ('__file__','__name__','OUT','HERE')})
import diffusers
source=ns
assert weights==original['weights'] and adapters==original['adapter']
def as_t(t):return torch.tensor(t['values'],dtype=torch.float32).reshape(t['shape'])
def dist(a,b):return float((a.float()-b.float()).abs().max())

def perturb(y,n,profile,step,branch):
    # Fixed algorithm declared independently of native values. Perturb source final
    # BF16 network prediction, not internal source math. Bounds recorded at each eval.
    z=y.to(torch.bfloat16)
    if profile=='opposed_up': signs=[branch=='conditional']*z.numel()
    elif profile=='opposed_down': signs=[branch!='conditional']*z.numel()
    else: signs=[hashlib.sha256(f'{profile}:{step}:{branch}:{i}'.encode()).digest()[0]>=128 for i in range(z.numel())]
    signs=torch.tensor(signs,dtype=torch.bool).reshape(z.shape)
    dest=torch.where(signs,torch.full_like(z,float('inf')),torch.full_like(z,float('-inf')))
    for _ in range(n):z=torch.nextafter(z,dest)
    return z.float()

def run(ed,rr,expected=None,mutation=None,ulp=0,profile=None):
    schedule=source['MOSSImageScheduler'](shift=5.0,training_shift=1.0)
    schedule.set_timesteps(2,device='cpu')
    x=hidden.clone().float(); trace=[]
    cache=ed.prepare_cache(rr) if mutation!='reference_uses_body_time' else None
    for i,tc in enumerate(schedule.timesteps):
        tnow=tc.float().reshape(1)/1000
        if mutation=='reference_uses_body_time':
            original_time=ed._time
            ed._time=lambda t,d:original_time(tnow if float(torch.as_tensor(t))==0 else t,d)
            cache=ed.prepare_cache(rr)
            ed._time=original_time
        c=predict(ed,x,tnow,rr,cache=cache).float()
        u=predict(ed,x,tnow,rr,negative,neg_mask,cache).float()
        c0=c.clone();u0=u.clone()
        if ulp:
            c=perturb(c,ulp,profile,i,'conditional');u=perturb(u,ulp,profile,i,'unconditional')
        vel=u+4.0*(c-u)
        dt=(schedule.sigmas[i+1]-schedule.sigmas[i]).float()
        out=schedule.step(vel,tc,x,return_dict=False)[0].float()
        manual=x+dt*vel
        item=dict(index=i,timestep=float(tc),sigma=float(schedule.sigmas[i]),next_sigma=float(schedule.sigmas[i+1]),dt=float(dt),manual_euler_max_abs=dist(out,manual),input=tensor(x),conditional=tensor(c),unconditional=tensor(u),guided=tensor(vel),output=tensor(out),prediction_perturbation=dict(conditional=dist(c,c0),unconditional=dist(u,u0)))
        # This is an exact same-input algebraic error bound, not a claim about trajectory sensitivity.
        item['same_input_cfg_euler_bound']=abs(float(dt))*(4*dist(c,c0)+3*dist(u,u0))
        if expected:
            e=expected['steps'][i]
            item['source_replay_errors']={k:dist(as_t(e[k]),v) for k,v in [('input',x),('conditional',c),('unconditional',u),('guided',vel),('output',out)]}
        trace.append(item);x=out
    return dict(final=tensor(x),steps=trace,sigmas=tensor(schedule.sigmas))

result=dict(schema=1,device='cpu',real_weights=False,native_data=False,parent_fixture_sha256=hashlib.sha256(parent_bytes).hexdigest(),source_sha256=original['source_sha256'],torch_version=torch.__version__,diffusers_version=diffusers.__version__,source_scheduler_file_sha256=hashlib.sha256(Path(inspect.getfile(FlowMatchEulerDiscreteScheduler)).read_bytes()).hexdigest(),perturbation_protocol='Prespecified nextafter on final source BF16 predictions: ±1 or ±2 representable values; profiles opposed_up, opposed_down, hash0..hash7; complete source re-evaluation at each perturbed latent. This is empirical sensitivity, not a universal bound.',cases=[])
with torch.no_grad():
    for count in [1,4]:
        rr=refs[:count];old=next(c for c in original['cases'] if c['reference_count']==count)
        correct=run(editor,rr,old['sampling'])
        assert all(v==0 for st in correct['steps'] for v in st['source_replay_errors'].values())
        assert dist(as_t(correct['final']),as_t(old['sampling']['final']))==0
        controls={}
        wrongtime=source['MossEditModel'](copy.deepcopy(model)).eval()
        bad=run(wrongtime,rr,mutation='reference_uses_body_time')
        controls['reference_uses_body_time']=dict(max_abs_difference=dist(as_t(bad['final']),as_t(correct['final'])),**bad)
        missing=source['MossEditModel'](copy.deepcopy(model)).eval()
        ob=missing._body_block
        missing._body_block=lambda layer,body,rope,mod,reference_kv=None,text_length=0:ob(layer,body,rope,mod,None,text_length)
        bad=run(missing,rr)
        controls['body_cannot_read_references']=dict(max_abs_difference=dist(as_t(bad['final']),as_t(correct['final'])),**bad)
        if count==4:
            bad=run(editor,list(reversed(rr)))
            controls['reversed_reference_order']=dict(max_abs_difference=dist(as_t(bad['final']),as_t(correct['final'])),**bad)
        trials=[]
        for n in [1,2]:
            for profile in ['opposed_up','opposed_down']+[f'hash{k}' for k in range(8)]:
                pert=run(editor,rr,ulp=n,profile=profile)
                trials.append(dict(ulp=n,profile=profile,final_max_abs_difference=dist(as_t(pert['final']),as_t(correct['final'])),**pert))
        bound=[]
        for st in correct['steps']:
            ec=(2/128)*float(as_t(st['conditional']).abs().max());eu=(2/128)*float(as_t(st['unconditional']).abs().max())
            bound.append(dict(index=st['index'],conditional_two_bf16_relative_eps_bound=ec,unconditional_two_bf16_relative_eps_bound=eu,same_input_cfg_euler_bound=abs(st['dt'])*(4*ec+3*eu)))
        result['cases'].append(dict(reference_count=count,correct=correct,full_sampler_controls=controls,predeclared_perturbation_trials=trials,conditional_two_epsilon_bound_sum_no_state_sensitivity=sum(x['same_input_cfg_euler_bound'] for x in bound),per_step_same_input_bounds=bound,existing_final_two_bf16_relative_eps_bound=(2/128)*float(as_t(correct['final']).abs().max())))
path=OUT/'analysis.json';path.write_text(json.dumps(result,separators=(',',':'))+'\n')
assert PARENT.read_bytes()==parent_bytes
receipt=dict(status='PASS_SOURCE_SAMPLER_REPLAY_AND_INDEPENDENT_CONTROLS',analysis_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_fixture_sha256=result['parent_fixture_sha256'],source_sha256=result['source_sha256'],source_scheduler_file_sha256=result['source_scheduler_file_sha256'],torch_version=torch.__version__,diffusers_version=diffusers.__version__,native_data=False,source_golden_unchanged=True,cases=[])
for c in result['cases']:
    receipt['cases'].append(dict(reference_count=c['reference_count'],sigmas=c['correct']['sigmas']['values'],controls={k:v['max_abs_difference'] for k,v in c['full_sampler_controls'].items()},perturbation_final_range={str(n):[min(t['final_max_abs_difference'] for t in c['predeclared_perturbation_trials'] if t['ulp']==n),max(t['final_max_abs_difference'] for t in c['predeclared_perturbation_trials'] if t['ulp']==n)] for n in [1,2]},existing_final_two_bf16_relative_eps_bound=c['existing_final_two_bf16_relative_eps_bound'],conditional_two_epsilon_bound_sum_no_state_sensitivity=c['conditional_two_epsilon_bound_sum_no_state_sensitivity'],source_replay_max_abs=0.0,manual_euler_max_abs=max(s['manual_euler_max_abs'] for s in c['correct']['steps'])))
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
